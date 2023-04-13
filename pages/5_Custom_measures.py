from typing import Union

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils import (
    draw_sidebar,
)


def to_rpn(expression: str) -> list[str]:
    """Converts an expression from infix notation to reverse polish notation.
    Operators and operands must be split by spaces.

    :param expression:
    :return:
    """

    operators_precedence = {"^": 4, "/": 3, "*": 3, "+": 2, "-": 2}
    operators_associativity = {
        "^": "right",
        "/": "left",
        "*": "left",
        "+": "left",
        "-": "left",
    }

    output_queue = []
    operator_stack = []

    current_token = ""

    for x in expression:
        if x.isalnum():
            current_token += x
            continue
        else:
            if current_token:
                output_queue.append(current_token)
                current_token = ""

        if x in operators_precedence:
            o1 = x
            o1_precedence = operators_precedence[o1]
            o1_associativity = operators_associativity[o1]
            for i, o2 in reversed(list(enumerate(operator_stack))):
                if o2 == "(":
                    break
                o2_precedence = operators_precedence[o2]
                if o2_precedence > o1_precedence or (
                    o1_precedence == o2_precedence and o1_associativity == "left"
                ):
                    output_queue.append(o2)
                    operator_stack.pop(i)
            operator_stack.append(o1)
        elif x == "(":
            operator_stack.append(x)
        elif x == ")":
            while True:
                o2 = operator_stack[-1]
                if o2 == "(":
                    operator_stack.pop(-1)
                    break

                output_queue.append(o2)
                operator_stack.pop(-1)

    if current_token:
        operator_stack.append(current_token)

    for x in list(reversed(operator_stack.copy())):
        # If the operator token on the top of the stack is a parenthesis, then there are mismatched parentheses.
        output_queue.append(x)

    return output_queue


def evaluate_rpn(tokens: list[Union[str, int, float]]) -> Union[int, float]:
    """Evaluate a list of tokens in Reverse Polish Notation.

    :param tokens: list of tokens in Reverse Polish Notation
    :return:
    """
    stack = []

    for token in tokens:
        if isinstance(token, (int, float)):
            # If the token is a number, push it onto the stack
            stack.append(token)
        elif token in ["+", "-", "*", "/"]:
            if token == "+":
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = operand1 + operand2
            elif token == "-":
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = operand1 - operand2
            elif token == "*":
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = operand1 * operand2
            elif token == "/":
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = operand1 / operand2
            # Push the result onto the stack
            stack.append(result)

    # The final result is the top element on the stack
    return stack[0]


def replace_rpn_tokens_with_numbers(rpn: list[str], token_dict: dict) -> list:
    """Evaluate items in Reverse Polish Notation and check if it is not an operator for its value in token_dict

    :param rpn: list of tokens in Reverse Polish Notation
    :param token_dict: dictionary with token names as keys and their values as values
    :return: list of tokens in Reverse Polish Notation with numbers instead of tokens
    """
    out_rpn = []
    for token in rpn:
        try:
            token = float(token)
            out_rpn.append(token)
        except ValueError:
            if token in {"+", "-", "*", "/"}:
                out_rpn.append(token)
            else:
                out_rpn.append(token_dict[token.lower()])

    return out_rpn


def rpn_to_latex(tokens: list) -> str:
    """Converts a list of tokens in Reverse Polish Notation to a LaTeX string.

    :param tokens: list of tokens in Reverse Polish Notation
    :return:
    """
    stack = []

    for token in tokens:
        if token in ["+", "-", "*", "/", "^"]:
            if token == "+":
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = f"{operand1} + {operand2}"
            elif token == "-":
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = f"{operand1} - {operand2}"
            elif token == "*":
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = f"{operand1} \cdot {operand2}"
            elif token == "/":
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = f"\\frac{{{operand1}}}{{{operand2}}}"
            elif token == "^":
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = f"{operand1} ^ {operand2}"
            stack.append(result)
        else:
            stack.append(token)

    return stack[0]


e, i, dataset_size = draw_sidebar()

st.title("Custom evaluation measures")

st.write(
    "This page allows you to create your own evaluation measures. "
    "Create the equation using the confusion matrix terms. "
    "Currently, the following operators are supported: `+`, `-`, `*`, `/` and parenthesis `()`. "
    "The following variables are predefined (case insensitive): "
    "`TN`, `TP`, `FP`, `FN`, `I` (total number of relevant documents), `E` (total number of irrelevant documents), "
    "`N` (total number of documents), `recall`, `precision`, `accuracy`."
    "\n"
    "There are three steps to visualise your own evaluation measure:\n"
    " 1. Create the equation (or select one from four predefined ones)\n"
    " 2. Define the x-axis\n"
    " 3. Define the y-axis\n"
    "Additionally, you can define the dataset size and the number of relevant documents using the sidebar."
)
st.markdown("---")

input_equation_before_text = "1. Type your equation here and press Enter"
input_equation = st.empty()
equation_string = input_equation.text_input(
    input_equation_before_text, value="(5*TP + TN)/N"
)

col1, col2, col3, col4 = st.columns(4)

if wss_button := col1.button("Work Saved over Sampling"):
    equation_string = "(TN + FN)/N - 1 + recall"
    input_equation.text_input(input_equation_before_text, value=equation_string)

if f1_button := col2.button("F1 score"):
    equation_string = "(2*TP)/(2*TP + FP + FN)"
    input_equation.text_input(input_equation_before_text, value=equation_string)

if nlr_button := col3.button("Negative likelihood ratio"):
    equation_string = "(FN * E)/(TN * I)"
    input_equation.text_input(input_equation_before_text, value=equation_string)

if mk_button := col4.button("Markedness"):
    equation_string = "((TP/(TP+FP) + TN/(FN+TN)) - 1)"
    input_equation.text_input(input_equation_before_text, value=equation_string)

rpn = to_rpn(equation_string)
latex_formula = rpn_to_latex(rpn)


df_3d = pd.DataFrame()
all_recalls = np.linspace(0.01, 1, 35)

for recall in all_recalls:
    TP = recall * i
    FN = (1 - recall) * i

    TN = np.linspace(1, e, 80)
    df = pd.DataFrame()
    for tn_score in TN:
        tn_score = int(tn_score)
        FP = e - tn_score

        token_dict = {
            "tp": TP,
            "fn": FN,
            "tn": tn_score,
            "fp": FP,
            "i": i,
            "e": e,
            "n": dataset_size,
            "recall": recall,
            "precision": TP / (TP + FP),
            "accuracy": (TP + tn_score) / dataset_size,
        }

        substituted_rpn = replace_rpn_tokens_with_numbers(rpn, token_dict=token_dict)
        score = evaluate_rpn(tokens=substituted_rpn)

        token_dict["measure"] = score

        df_3d = df_3d.append(
            token_dict,
            ignore_index=True,
        )

dimensions = [
    "Recall",
    "Precision",
    "Accuracy",
    "TP",
    "TN",
    "FP",
    "FN",
]
dim_x_col, dim_y_col = st.columns(2)
dimension_x = dim_x_col.selectbox("2. Select X axis measure: ", dimensions, key="dim_x")
dimension_y = dim_y_col.selectbox(
    "3. Select Y axis measure: ",
    [x for x in dimensions if x != dimension_x],
    key="dim_y",
    index=3,
)
st.markdown("---")
st.write(f"3D plot of ${dimension_x}$ vs ${dimension_y}$ vs ${latex_formula}$")

fig = px.scatter_3d(
    df_3d,
    x=dimension_x.lower(),
    y=dimension_y.lower(),
    z="measure",
    color="measure",
    opacity=0.7,
    width=800,
    height=600,
)
fig.update_layout(
    scene=dict(
        xaxis_title=dimension_x,
        yaxis_title=dimension_y,
        zaxis_title="measure",
    ),
)
st.plotly_chart(fig)
