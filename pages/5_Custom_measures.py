import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils import (
    draw_sidebar,
)


def to_rpn(expression):
    """Operators and operands must be split by spaces."""

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


def evaluate_rpn(tokens):
    # Create a stack to store the operands
    stack = []

    # Iterate through the tokens
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
    # evaluate items in rpn and check if it is not an operator for its value in token_dict
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


def rpn_to_latex(tokens: list):
    # Create a stack to store operands
    stack = []

    # Iterate through the tokens
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
            # Push the result onto the stack
            stack.append(result)
        else:
            stack.append(token)

    return stack[0]


e, i, dataset_size = draw_sidebar()

st.title("Custom evaluation measures")

st.write(
    "This page allows you to create your own evaluation measures. "
    "Create the equation using the confusion matrix terms."
    "Currently, the following operators are supported: `+`, `-`, `*`, `/` and parenthesis. "
    "The following variables are predefined (case insensitive): "
    "`TN`, `TP`, `FP`, `FN`, `I` (total number of relevant documents), `E` (total number of irrelevant documents), "
    "`N` (total number of documents), `recall`, `precision`, `accuracy`."
)

input_equation = st.empty()
equation_string = input_equation.text_input(
    "Type your equation here and press Enter", value="(5*TP + TN)/N"
)

col1, col2, col3, col4 = st.columns(4)

if wss_button := col1.button("Work Saved over Sampling"):
    equation_string = "(TN + FN)/N - 1 + recall"
    input_equation.text_input(
        "Type your equation here and press Enter", value=equation_string
    )

if f1_button := col2.button("F1 score"):
    equation_string = "(2*TP)/(2*TP + FP + FN)"
    input_equation.text_input(
        "Type your equation here and press Enter", value=equation_string
    )

if nlr_button := col3.button("Negative likelihood ratio"):
    equation_string = "(FN * E)/(TN * I)"
    input_equation.text_input(
        "Type your equation here and press Enter", value=equation_string
    )

if mk_button := col4.button("Markedness"):
    equation_string = "((TP/(TP+FP) + TN/(FN+TN)) - 1)"
    input_equation.text_input(
        "Type your equation here and press Enter", value=equation_string
    )

rpn = to_rpn(equation_string)
latex_formula = rpn_to_latex(rpn)
st.latex(latex_formula)

df_3d = pd.DataFrame()
all_recalls = np.linspace(0.01, 1, 25)

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

fig = px.scatter_3d(
    df_3d,
    x="tn",
    y="recall",
    z="measure",
    color="measure",
    opacity=0.7,
    width=800,
    height=600,
)
fig.update_layout(
    scene=dict(
        xaxis_title="TN",
        yaxis_title="recall",
        zaxis_title="measure",
    ),
)
st.plotly_chart(fig)
