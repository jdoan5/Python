import sys, yaml
def emit(node, indent=2):
    if isinstance(node, dict):
        for k, v in node.items():
            print("  "*indent + str(k))
            emit(v, indent+1)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            print("  "*indent + f"- {i}")
            emit(v, indent+1)
    else:
        print("  "*indent + str(node))

path = sys.argv[1] if len(sys.argv) > 1 else "dq_config.yaml"
with open(path, "r") as f: data = yaml.safe_load(f)

print("```mermaid")
print("mindmap")
print("  root((config))")
emit(data, 2)
print("```")