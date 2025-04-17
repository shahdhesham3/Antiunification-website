from flask import Flask, render_template, request, jsonify
import networkx as nx
import matplotlib.pyplot as plt
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'your_project'))

# Initialize the GPT-2 generator
generator = pipeline('text-generation', model='gpt2', framework='pt')

# Ask AI Anything using Hugging Face model
def ask_ai_anything_huggingface(question):
    try:
        model_name = "EleutherAI/gpt-neo-1.3B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        inputs = tokenizer.encode(question, return_tensors="pt")
        outputs = model.generate(inputs, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        return f"Error using Hugging Face model: {str(e)}"

# Handle and parse a predicate logic expression
def handle_predicate_logic(expr):
    if isinstance(expr, tuple):
        if expr[0] == '∀':
            variable = expr[1]
            sub_expr = expr[2]
            sentence = f"For all {variable}, {handle_predicate_logic(sub_expr)}."
            return sentence
        elif expr[0] == '∃':
            variable = expr[1]
            sub_expr = expr[2]
            sentence = f"There exists {variable} such that {handle_predicate_logic(sub_expr)}."
            return sentence
        elif is_predicate(expr[0]):
            predicate = expr[0]
            args = ', '.join(map(str, expr[1:]))
            return f"{predicate}({args})"
        elif expr[0] == '→':
            left = handle_predicate_logic(expr[1])
            right = handle_predicate_logic(expr[2])
            return f"({left} → {right})"
        elif expr[0] == '∧':
            left = handle_predicate_logic(expr[1])
            right = handle_predicate_logic(expr[2])
            return f"({left} ∧ {right})"
    return str(expr)

# Check if a symbol is a predicate (e.g., P, Q)
def is_predicate(symbol):
    return isinstance(symbol, str) and symbol.isalpha()

# Generalize two expressions (LGG)
def generalize(expr1, expr2):
    if expr1 is None or expr2 is None:
        return None

    if expr1 == expr2:
        return expr1

    if not isinstance(expr1, tuple) or not isinstance(expr2, tuple):
        return ('_', '_')

    if expr1[0] != expr2[0]:
        return '_'

    len1, len2 = len(expr1) - 1, len(expr2) - 1
    max_len = max(len1, len2)

    generalized_args = []
    for i in range(max_len):
        if i < len1 and i < len2:
            generalized_args.append(generalize(expr1[i + 1], expr2[i + 1]))
        else:
            generalized_args.append('_')

    return (expr1[0],) + tuple(generalized_args)

# Draw tree representation of the expression
def draw_tree(expression, title="Expression Tree"):
    if expression is None:
        return

    G = nx.DiGraph()

    def add_nodes_edges(expr, parent=None):
        if isinstance(expr, tuple):
            node_label = expr[0]
            node_id = f"{node_label}_{id(expr)}"
            G.add_node(node_id, label=node_label)

            if parent:
                G.add_edge(parent, node_id)

            for sub_expr in expr[1:]:
                add_nodes_edges(sub_expr, node_id)
        else:
            G.add_node(expr, label=expr)
            if parent:
                G.add_edge(parent, expr)

    add_nodes_edges(expression)

    pos = nx.spring_layout(G)
    labels = {node: G.nodes[node]['label'] for node in G.nodes}

    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color="lightblue", font_size=10, edge_color="gray")
    plt.title(title)
    plt.show()

# Visualize predicate logic with a sentence
def visualize_predicate_logic_with_sentence(expr):
    draw_tree(expr, "Predicate Logic Expression Tree")
    predicate_sentence = handle_predicate_logic(expr)
    print(f"Predicate Sentence: {predicate_sentence}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visual_tool')
def visual_tool():
    return render_template('tool.html')

@app.route('/generate_text', methods=['POST'])
def generate_text():
    input_text = request.form['input_text']
    response = generator(input_text, max_length=50)
    return jsonify({'generated_text': response[0]['generated_text']})

@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    question = request.form['question']
    response = ask_ai_anything_huggingface(question)
    return jsonify({'ai_response': response})

@app.route('/parse_expression', methods=['POST'])
def parse_expression():
    user_input = request.form['expression']
    try:
        expr = ast.literal_eval(user_input)
        parsed_expr = handle_predicate_logic(expr)
        return jsonify({'parsed_expression': parsed_expr})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/generalize_expressions', methods=['POST'])
def generalize_expressions():
    expr1_input = request.form['expr1']
    expr2_input = request.form['expr2']
    expr1 = ast.literal_eval(expr1_input)
    expr2 = ast.literal_eval(expr2_input)
    generalized_expr = generalize(expr1, expr2)
    return jsonify({'generalized_expression': generalized_expr})

if __name__ == "__main__":
    app.run(debug=True)
