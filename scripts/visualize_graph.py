import argparse
import sqlite3
import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import settings

def serve_reports():
    report_dir = settings.paths.reports
    if not os.path.exists(report_dir):
        print(f"[!] Report directory {report_dir} does not exist.")
        return
    os.chdir(report_dir)
    port = 8000
    server_address = ("", port)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    url = f"http://localhost:{port}/attack_graph.html"
    print(f"[-] Serving reports at {url}")
    webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[-] Server stopped.")

def inspect_node(node_id, depth=1):
    db_path = settings.paths.database
    if not os.path.exists(db_path):
        print("[!] Database not found.")
        return
    try:
        import networkx as nx
        from pyvis.network import Network
    except ImportError:
        print("[!] Please install pyvis: pip install pyvis")
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    G = nx.DiGraph()
    queue = [(node_id, 0)]
    visited = set()
    while queue:
        current, current_depth = queue.pop(0)
        if current in visited or current_depth > depth: continue
        visited.add(current)
        cursor.execute("SELECT name, description FROM nodes WHERE id=?", (current,))
        row = cursor.fetchone()
        label = f"{current}\n{row[0]}" if row else current
        G.add_node(current, label=label, color="#ff9999" if current == node_id else "#97c2fc")
        cursor.execute("SELECT target, relation, probability FROM edges WHERE source=?", (current,))
        edges = cursor.fetchall()
        for target, relation, prob in edges:
            G.add_edge(current, target, label=f"{relation}\n({prob:.2f})")
            if target not in visited: queue.append((target, current_depth + 1))
    conn.close()
    if len(G.nodes) == 0:
        print(f"[!] Node {node_id} not found.")
        return
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    net.from_nx(G)
    output_file = f"inspection_{node_id}.html"
    net.save_graph(output_file)
    target_path = os.path.join(settings.paths.reports, output_file)
    if os.path.abspath(output_file) != os.path.abspath(target_path):
        os.rename(output_file, target_path)
    print(f"[-] Graph saved to {target_path}")
    webbrowser.open("file://" + os.path.abspath(target_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve")
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("id")
    inspect_parser.add_argument("--depth", type=int, default=1)
    args = parser.parse_args()
    if args.command == "serve": serve_reports()
    elif args.command == "inspect": inspect_node(args.id, args.depth)
