from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CalcInput:
    name: str
    value: float
    unit: str
    chunk_id: str = ""


@dataclass
class CalcResult:
    formula: str
    inputs: list[CalcInput]
    result: float
    rounding: str = "one decimal place"

    def to_dict(self) -> dict:
        return {
            "formula": self.formula,
            "inputs": [
                {
                    "name": i.name,
                    "value": i.value,
                    "unit": i.unit,
                    "chunk_id": i.chunk_id,
                }
                for i in self.inputs
            ],
            "result": self.result,
            "rounding": self.rounding,
        }


def percentage_change(new: float, old: float) -> CalcResult:
    if old == 0:
        raise ValueError("Cannot calculate percentage change from zero base")

    result = (new - old) / abs(old) * 100

    return CalcResult(
        formula=rf"\frac{{{new} - {old}}}{{|{old}|}} \times 100",
        inputs=[
            CalcInput(name="new_value", value=new, unit=""),
            CalcInput(name="old_value", value=old, unit=""),
        ],
        result=round(result, 1),
    )


def margin(
    numerator: float,
    denominator: float,
    metric_name: str = "margin",
) -> CalcResult:
    if denominator == 0:
        raise ValueError(f"Denominator cannot be zero for {metric_name}")

    result = numerator / denominator * 100

    return CalcResult(
        formula=rf"\frac{{{numerator}}}{{{denominator}}} \times 100",
        inputs=[
            CalcInput(
                name=f"{metric_name}_numerator",
                value=numerator,
                unit="",
            ),
            CalcInput(
                name=f"{metric_name}_denominator",
                value=denominator,
                unit="",
            ),
        ],
        result=round(result, 1),
    )


def gross_margin(revenue: float, cost_of_revenue: float) -> CalcResult:
    if revenue == 0:
        raise ValueError("Revenue cannot be zero for gross margin")

    gross_profit = revenue - cost_of_revenue

    return CalcResult(
        formula=rf"\frac{{{revenue} - {cost_of_revenue}}}{{{revenue}}} \times 100",
        inputs=[
            CalcInput(name="revenue", value=revenue, unit=""),
            CalcInput(
                name="cost_of_revenue",
                value=cost_of_revenue,
                unit="",
            ),
        ],
        result=round(gross_profit / revenue * 100, 1),
    )


def operating_margin(
    operating_income: float,
    revenue: float,
) -> CalcResult:
    return margin(operating_income, revenue, "operating_margin")


def net_margin(
    net_income: float,
    revenue: float,
) -> CalcResult:
    return margin(net_income, revenue, "net_margin")


def ratio(
    numerator: float,
    denominator: float,
    ratio_name: str = "ratio",
) -> CalcResult:
    if denominator == 0:
        raise ValueError(f"Denominator cannot be zero for {ratio_name}")

    result = numerator / denominator

    return CalcResult(
        formula=rf"\frac{{{numerator}}}{{{denominator}}}",
        inputs=[
            CalcInput(
                name=f"{ratio_name}_numerator",
                value=numerator,
                unit="",
            ),
            CalcInput(
                name=f"{ratio_name}_denominator",
                value=denominator,
                unit="",
            ),
        ],
        result=round(result, 2),
    )


def absolute_change(new: float, old: float) -> CalcResult:
    result = new - old

    return CalcResult(
        formula=rf"{new} - {old}",
        inputs=[
            CalcInput(name="new_value", value=new, unit=""),
            CalcInput(name="old_value", value=old, unit=""),
        ],
        result=round(result, 1),
    )


CALCULATION_REGISTRY = {
    "percentage_change": percentage_change,
    "margin": margin,
    "gross_margin": gross_margin,
    "operating_margin": operating_margin,
    "net_margin": net_margin,
    "ratio": ratio,
    "absolute_change": absolute_change,
}
