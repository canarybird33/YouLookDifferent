# encoding: utf-8

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine

from utils.reid_metric import R1_mAP, R1_mAP_reranking, R1_mAP_longterm, R1_mAP_reranking_longterm


def create_supervised_evaluator(model, metrics,
                                device=None, write_feat = False):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids, clothid = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids, clothid

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        cfg,
        model,
        val_loader,
        num_query
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("TEST clothing change re-id")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        evaluator1 = create_supervised_evaluator(model, metrics={'r1_mAP_longterm': R1_mAP_longterm(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
        evaluator2 = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
        evaluator = (evaluator1, evaluator2)

    elif cfg.TEST.RE_RANKING == 'yes':
        print("Create evaluator for reranking")
        evaluator1 = create_supervised_evaluator(model, metrics={'r1_mAP_longterm': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
        evaluator2 = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
        evaluator = (evaluator1, evaluator2)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))

    evaluator[0].run(val_loader)
    CC_cmc, CC_mAP = evaluator[0].state.metrics['r1_mAP_longterm']
    logger.info('>>>>> TEST: Cloth changing evaluation results:')
    logger.info("mAP: {:.1%}".format(CC_mAP))
    for r in [1, 5, 10, 20, 50]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, CC_cmc[r - 1]))

    evaluator[1].run(val_loader)
    SS_cmc, SS_mAP = evaluator[1].state.metrics['r1_mAP']
    logger.info('>>>>> TEST: Standard evaluation results:')
    logger.info("mAP: {:.1%}".format(SS_mAP))
    for r in [1, 5, 10, 20, 50]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, SS_cmc[r - 1]))

    return CC_cmc, CC_mAP, SS_cmc, SS_mAP