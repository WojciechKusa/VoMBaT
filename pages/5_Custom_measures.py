import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils import (
    get_dataset_parameters,
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
        # if x not in operators_precedence and x not in ["(", ")"] and not " ":  # Is number
        # if c.isalnum():
        #     If the character is alphanumeric, add it to the current token
        if x.isalnum():
            current_token += x
            # output_queue.append(x)
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
            # If the token is an operator, pop the required operands from the stack
            # and perform the operation
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
            if token in ["+", "-", "*", "/"]:
                out_rpn.append(token)
            else:
                out_rpn.append(token_dict.get(token, 0))

    return out_rpn


with open("data/datasets.json", "r") as f:
    datasets = json.load(f)

# Sidebar
st.sidebar.write("### Dataset parameters")

emp = st.sidebar.empty()
dataset_type = emp.selectbox(
    label="Select a dataset type", options=datasets.keys(), key="dataset_picker"
)

if dataset_type == "Custom":
    _dataset_size = st.session_state.dataset_size
    _i_percentage = st.session_state.i_percentage
    _i = int(_dataset_size * _i_percentage / 100)
    _e = _dataset_size - _i
else:
    _dataset_size, _i, _e, _i_percentage = get_dataset_parameters(
        dataset_type=dataset_type
    )


def check_dataset_size():
    if _dataset_size != st.session_state.dataset_size:
        st.session_state["dataset_picker"] = "Custom"


def check_i_percentage():
    if _i_percentage != st.session_state.i_percentage:
        st.session_state["dataset_picker"] = "Custom"


dataset_size = st.sidebar.slider(
    "Dataset size",
    100,
    5000,
    _dataset_size,
    50,
    key="dataset_size",
    on_change=check_dataset_size,
)
i_percentage = st.sidebar.slider(
    "Percentage of relevant documents (includes)",
    1.0,
    99.0,
    _i_percentage,
    1.0,
    key="i_percentage",
    on_change=check_i_percentage,
)

i = int(dataset_size * i_percentage / 100)
e = dataset_size - i
st.sidebar.write("Number of relevant documents (includes): ", i)
st.sidebar.write("Number of non-relevant documents (excludes): ", e)
# add an input field so user can type their own evaluation measure
# add a dropdown menu so user can select from a list of evaluation measures


st.title("Custom evaluation measures")

st.write(
    "This page allows you to create your own evaluation measures. "
    "Create the equation using the confusion matrix terms."
    "Currently, the following operators are supported: `+`, `-`, `*`, `/` and parenthesis. "
    "The following variables are supported (case sensitive): "
    "`TN`, `TP`, `FP`, `FN`, `i`, `e`, `N`, `recall`, `precision`, `accuracy`."
)

input_equation = st.empty()
equation_string = input_equation.text_input(
    "Type your equation here and press Enter", value="(TP + TN)/N"
)

col1, col2 = st.columns(2)

if wss_button := col1.button("WSS"):
    equation_string = "(TN + FN)/N - (1 - recall)"
    input_equation.text_input(
        "Type your equation here and press Enter", value=equation_string
    )

if f1_button := col2.button("F1 score"):
    equation_string = "(2*TP)/(2*TP + FP + FN)"
    input_equation.text_input(
        "Type your equation here and press Enter", value=equation_string
    )


recall = st.slider("Estimated recall: ", 1, 100, 95, 1)
recall /= 100

# for recall in all_recalls:
TP = recall * i
FN = (1 - recall) * i


TN = np.array(range(e + 1))
df = pd.DataFrame()
for tn_score in TN:
    tn_score = int(tn_score)
    # TN = np.array(range(e + 1))
    FP = e - tn_score

    # create a dictionary of the confusion matrix terms
    token_dict = {
        "TP": TP,
        "FN": FN,
        "TN": tn_score,
        "FP": FP,
        "i": i,
        "e": e,
        "N": dataset_size,
        "recall": recall,
        "precision": TP / (TP + FP),
        "accuracy": (TP + tn_score) / dataset_size,
    }

    rpn = to_rpn(equation_string)
    # st.write(rpn)
    rpn = replace_rpn_tokens_with_numbers(rpn, token_dict=token_dict)
    # st.write(rpn)
    score = evaluate_rpn(tokens=rpn)

    # st.write(f"TN: {TN}, score: {score}")
    token_dict["measure"] = score

    df = df.append(
        token_dict,
        ignore_index=True,
    )

fig = px.line(
    df,
    x="TN",
    y="measure",
    width=800,
    height=450,
)
fig.update_layout(
    xaxis_title=r"TN",
    yaxis_title=r"Evaluation measure score",
    legend_title_text="Measures",
)
st.plotly_chart(fig)
