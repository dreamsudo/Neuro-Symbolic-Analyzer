import os
import json
from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)

class GraphVisualizer:
    def __init__(self, worlds):
        self.worlds = worlds
        self.output_dir = settings.paths.reports

    def generate_html(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        nodes = []
        edges = []
        
        for i, world in enumerate(self.worlds):
            world_node_id = f"World_{i}"
            active_threats = [f"{k} ({v:.2f})" for k, v in world.facts.items() if v > 0.5]
            label = f"World {i}\n" + "\n".join(active_threats)
            
            nodes.append({"id": world_node_id, "label": label, "color": "#97c2fc", "shape": "box"})
            if i > 0:
                edges.append({"from": f"World_{i-1}", "to": world_node_id, "arrows": "to", "label": "Transition"})

        data = {"nodes": nodes, "edges": edges}
        json_data = json.dumps(data)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Neurosymbolic Attack Graph</title>
            <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
            <style type="text/css">#mynetwork {{ width: 100%; height: 600px; border: 1px solid lightgray; }}</style>
        </head>
        <body>
            <h2>Epistemological Permutations (Fuzzy Logic)</h2>
            <div id="mynetwork"></div>
            <script type="text/javascript">
                var data = {json_data};
                var container = document.getElementById('mynetwork');
                var options = {{ layout: {{ hierarchical: {{ direction: "LR", sortMethod: "directed" }} }}, physics: false }};
                var network = new vis.Network(container, data, options);
            </script>
        </body>
        </html>
        """
        
        output_path = os.path.join(self.output_dir, "attack_graph.html")
        with open(output_path, "w") as f:
            f.write(html_content)
        logger.info(f"Graph visualization generated at: {output_path}")
