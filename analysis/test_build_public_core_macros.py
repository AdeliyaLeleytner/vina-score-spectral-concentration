from __future__ import annotations

import json

import build_public_core_macros as macros


def test_released_macros_are_exactly_generated_from_public_core() -> None:
    ledger = json.loads(macros.DEFAULT_INPUT.read_text())
    expected = macros.render(ledger, macros.sha256_file(macros.DEFAULT_INPUT))
    assert macros.DEFAULT_OUTPUT.read_text() == expected


def test_headline_macro_values_are_locked() -> None:
    ledger = json.loads(macros.DEFAULT_INPUT.read_text())
    values = dict(macros.macro_values(ledger))
    assert values["DffResidualPR"] == "9.303"
    assert values["DsResidualPR"] == "18.206"
    assert values["DffMWMatchedControlRho"] == "0.992"
    assert values["DsMWMatchedControlRho"] == "0.994"
    assert values["DffSizeAxisRhoRange"] == "0.200--0.207"
    assert values["DsSizeAxisRhoRange"] == "0.322--0.361"
    assert values["DffScaffoldProbeTwoHundred"] == "0.925"
    assert values["DffScaffoldProbeFiveHundred"] == "0.958"
    assert values["DsScaffoldProbeTwoHundred"] == "0.914"
    assert values["DsScaffoldProbeFiveHundred"] == "0.958"
    assert values["DffMWConditionedNullPR"] == "34.875"
    assert values["DsMWConditionedNullPR"] == "47.039"
    assert values["DffMWConditionedExplainedShare"] == "0.238"
    assert values["DsMWConditionedExplainedShare"] == "0.251"
    assert values["DffMWConditionedMapRho"] == "0.924"
    assert values["DsMWConditionedMapRho"] == "0.718"
    assert values["DffMWConditionedSignAgreement"] == "0.834"
    assert values["DsMWConditionedSignAgreement"] == "0.727"
    assert values["DffRandomHoldoutTwoHundred"] == "0.935"
    assert values["DffRandomHoldoutFiveHundred"] == "0.968"
    assert values["DsRandomHoldoutTwoHundred"] == "0.917"
    assert values["DsRandomHoldoutFiveHundred"] == "0.960"
    assert values["DffSelectorRawRtwo"] == "0.900"
    assert values["DsSelectorRawRtwo"] == "0.698"
    assert values["DffSelectorResidualRtwo"] == "0.278"
    assert values["DsSelectorResidualRtwo"] == "0.219"
    assert values["DffSelectorDirectResidualRtwo"] == "0.278"
    assert values["DsSelectorDirectResidualRtwo"] == "0.219"
    assert values["DffSelectorResidualPCOneRtwo"] == "0.180"
    assert values["DsSelectorResidualPCOneRtwo"] == "0.065"
    assert values["DffSelectorDirectResidualDelta"] == "0.000015"
    assert values["DsSelectorDirectResidualDelta"] == "-0.000001"
    assert values["DffSelectorCommonResidualDelta"] == "0.036"
    assert values["DsSelectorCommonResidualDelta"] == "0.058"
    assert values["ExperimentalCrossRawRho"] == "0.582"
    assert values["HotspotLigands"] == "176"
    assert values["HotspotTargets"] == "21"
    assert values["HotspotSplitRho"] == "0.794"
    assert values["DavisHotspotRawRho"] == "0.690"
    assert values["DavisHotspotResidualRho"] == "0.765"
    assert values["PkistwoHotspotRawRho"] == "0.526"
    assert values["PkistwoHotspotResidualRho"] == "0.751"
    assert values["PanelTransferPositiveCells"] == "14"
    assert values["PanelTransferTotalCells"] == "15"
    assert values["PanelTransferConservativeMean"] == "0.0607"
    assert values["PanelTransferJointAUC"] == "0.0616"
    assert values["PanelTransferJointP"] == "0.0069"
    assert values["PanelRankPositiveCells"] == "14"
    assert values["PanelRankConservativeMean"] == "0.0384"
    assert values["PanelRankP"] == "0.0056"
    assert values["PanelTargetLooPositive"] == "21"
    assert values["PanelTargetLooTotal"] == "21"
    assert values["PanelTargetLooAucLow"] == "0.0294"
    assert values["PanelTargetLooAucHigh"] == "0.0622"
    assert values["PanelTargetLooCellWinsLow"] == "11"
    assert values["PanelTargetLooCellWinsHigh"] == "15"
    assert values["HotspotClusterPositiveFraction"] == "1.000"
    assert values["ExperimentalCrossQAP"] == "0.0001"
    assert len(values) == 119
