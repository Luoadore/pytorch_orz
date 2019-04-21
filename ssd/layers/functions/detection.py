import torch
import torch.autograd as Function
from ..box_utils import decode, nms

class Detect(Function):
    """
    At test time, Detect is the final layer of SSD, Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k/max_per_image number of output predictions for both
    confidence score and location.
    """
    def __init__(self, num_calsses, bkg_label, cfg, max_per_image, conf_thresh, nms_thresh):
        self.num_classes = num_calsses
        self.background_label = bkg_label
        self.max_per_image = max_per_image
        # parameters used in nms
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold nust be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data):
        """

        :param loc_data:(tensor) Loc preds from loc layers
                        Shape:[batch, num_priors*4]
        :param conf_data:(tensor) Conf preds from conf layers
                        Shape:[batch*num_priors, num_classes]
        :param prior_data:(tensor) Prior boxes and variances from priorbox layers
                        Shape:[1,num_priors, 4]
        :return:
        """
        num = loc_data.size(0) # batch size
        num_priors = prior_data.size(0)
        # top_k is 200 by defalut, num is 1 when testing because image input one by one
        output = torch.zeros(num, self.num_classes, self.max_per_image, 5)
        # for each sample, every prior box will have #num_classes conf. scorces for it
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        # Decode predictionbs into bboxes based on loc_data(offsets) and prior_data
        for i in range(num): # for each batch
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # for each class, perform nms to get bbox for this class
            conf_scores = conf_preds[i].clone() # conf_scores = (num_classes, num_priors)

            for cl in range(1, self.num_classes):
                # for particular class, keep those boxes with score greater than threshold
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                # scores are those higher than threshold
                scores = conf_scores[cl][c_mask]
                # for this class, no object of this class exists in this image, because all scores < threshold
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # step 1.remaining boxes for reasonable classes's objects, boxes being kept after conf_thresh
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # setp2. use NMS to removeredundant boxes bounding the same class's object
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.max_per_image)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)

        # TODO: what flt doing
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort()
        flt[(rank < self.max_per_image).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output # before
        # return flt # after