import re
import sys
import argparse
import random
import os
from collections import defaultdict, deque
from heapq import heappop, heappush
from graphviz import Digraph
from math import log


class TextGraph:
    def __init__(self):
        self.graph = defaultdict(lambda: defaultdict(int))
        self.nodes = set()

    def process_text(self, text):
        """预处理文本：替换非字母字符为空格，转换为小写"""
        cleaned = re.sub(r'[^a-zA-Z]', ' ', text).lower()
        words = cleaned.split()
        return words

    def build_graph(self, words):
        """构建有向图"""
        for i in range(len(words) - 1):
            from_word = words[i]
            to_word = words[i + 1]
            self.nodes.add(from_word)
            self.nodes.add(to_word)
            self.graph[from_word][to_word] += 1

    def show_graph(self):
        """显示图结构"""
        print("有向图结构：")
        for from_node in sorted(self.graph.keys()):
            for to_node in sorted(self.graph[from_node].keys()):
                print(f"{from_node} -> {to_node} (权重: {self.graph[from_node][to_node]})")

    def shortest_path(self, word1, word2=None):
        """优化后的最短路径算法，支持单/双单词查询"""
        if word1 not in self.nodes:
            return None, float('inf'), []

        # 单单词模式：计算到所有节点的最短路径
        if word2 is None:
            results = {}
            for target in self.nodes:
                if target != word1:
                    path, dist, _ = self._dijkstra(word1, target)
                    if path:
                        results[target] = (path, dist)
            return None, None, results

        # 双单词模式
        if word2 not in self.nodes:
            return None, float('inf'), []

        if word1 == word2:
            return [word1], 0, [[word1]]

        return self._dijkstra(word1, word2)

    def _dijkstra(self, start, end):
        """改进的Dijkstra算法，记录所有最短路径"""
        heap = [(0, start, [])]
        visited = {}
        shortest_paths = []
        min_dist = float('inf')

        while heap:
            dist, node, path = heappop(heap)

            # 如果已经找到更短的路径，跳过
            if dist > min_dist:
                continue

            # 到达终点
            if node == end:
                if dist < min_dist:
                    shortest_paths = [path + [node]]
                    min_dist = dist
                elif dist == min_dist:
                    shortest_paths.append(path + [node])
                continue

            # 检查是否已经以更短距离访问过
            if node in visited and visited[node] < dist:
                continue

            visited[node] = dist
            new_path = path + [node]

            for neighbor, weight in self.graph[node].items():
                heappush(heap, (dist + weight, neighbor, new_path))

        if not shortest_paths:
            return None, float('inf'), []

        return shortest_paths[0], min_dist, shortest_paths

    def find_bridge_words(self, word1, word2):
        """查找从 word1 到 word2 的桥接词，输出指定格式"""

        word1 = word1.strip().lower()
        word2 = word2.strip().lower()
        if word1 not in self.nodes or word2 not in self.nodes:
            return []

        bridge_words = []
        for word3 in self.graph[word1]:
            if word2 in self.graph[word3]:
                bridge_words.append(word3)

        return bridge_words

    def generate_new_text(self, input_text):
        """根据桥接词生成新文本"""
        cleaned = re.sub(r'[^a-zA-Z]', ' ', input_text)
        original_words = cleaned.split()
        lower_words = [word.lower() for word in original_words]

        new_words = []

        for i in range(len(lower_words) - 1):
            word1 = lower_words[i]
            word2 = lower_words[i + 1]
            new_words.append(original_words[i])  # 保留原始大小写

            bridges = self.find_bridge_words(word1, word2)
            if bridges:
                bridge = random.choice(bridges)
                # 确保桥接词的首字母大写，与原始文本风格一致
                if original_words[i][0].isupper():
                    bridge = bridge.capitalize()
                new_words.append(bridge)

        new_words.append(original_words[-1])
        return ' '.join(new_words)

    def random_walk(self):
        """改进的随机游走，直到出现重复边或没有出边为止"""
        if not self.nodes:
            return [], 0, set()

        current = random.choice(list(self.nodes))
        path = [current]
        edges_visited = set()
        total_weight = 0
        stop_flag = False

        while True:
            if stop_flag or current not in self.graph or not self.graph[current]:
                break

            next_nodes = list(self.graph[current].items())
            next_node, weight = random.choice(next_nodes)

            edge = (current, next_node)
            if edge in edges_visited:
                break

            edges_visited.add(edge)
            path.append(next_node)
            total_weight += weight
            current = next_node

            print(f"当前路径: {' -> '.join(path)}")
            print(f"已遍历边: {len(edges_visited)}, 总权重: {total_weight}")
            cmd = input("按回车继续，输入'stop'停止: ").strip().lower()
            if cmd == 'stop':
                stop_flag = True

        return path, total_weight, edges_visited

    def save_walk_result(self, path, edges, filename="random_walk.txt"):
        """保存随机游走结果到文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("随机游走路径:\n")
                f.write(' '.join(path) + '\n\n')
                f.write("遍历的边:\n")
                for edge in edges:
                    f.write(f"{edge[0]} -> {edge[1]}\n")
                f.write(f"\n总边数: {len(edges)}\n")
            print(f"结果已保存到 {filename}")
        except Exception as e:
            print(f"保存文件时出错: {e}")

    def save_graph_image(self, filename="text_graph", format="png"):
        """
        使用Graphviz生成有向图图像并保存到文件
        参数:
            filename: 文件名(不带扩展名)
            format: 图像格式(png, pdf, svg等)
        返回: 生成的文件路径
        """
        try:
            os.environ["GRAPHVIZ_DOT"] = r"D://Graphviz-12.2.1-win64//bin//dot.exe"
            dot = Digraph(comment='Text Graph', engine='dot')
            dot.attr(rankdir='LR', size='20,20')

            for node in self.nodes:
                dot.node(node, shape='circle')

            for from_node in self.graph:
                for to_node in self.graph[from_node]:
                    weight = self.graph[from_node][to_node]
                    dot.edge(from_node, to_node, label=str(weight), fontsize='10')

            dot.attr('node', shape='circle', fontname='Arial')
            dot.attr('edge', fontname='Arial', fontsize='10')

            filepath = dot.render(filename=filename, format=format, cleanup=True)
            return filepath
        except Exception as e:
            print(f"生成图形时出错: {e}")
            return None

    def show_shortest_path_on_graph(self, word1, word2=None):
        """在图上展示最短路径"""
        if word1 not in self.nodes:
            print(f"提示: 图中不存在 {word1}")
            return

        if word2 is None:
            # 单单词模式
            results = self.shortest_path(word1)[2]
            for target, (path, dist) in results.items():
                print(f"从 {word1} 到 {target} 的最短路径: {' -> '.join(path)} (长度: {dist})")
                self._highlight_path_on_graph(path, dist, f"shortest_path_{word1}_to_{target}")
        else:
            if word2 not in self.nodes:
                print(f"提示: 图中不存在 {word2}")
                return
            path, dist, _ = self.shortest_path(word1, word2)
            if path:
                print(f"从 {word1} 到 {word2} 的最短路径: {' -> '.join(path)} (长度: {dist})")
                self._highlight_path_on_graph(path, dist, f"shortest_path_{word1}_to_{word2}")
            else:
                print(f"提示: 从 {word1} 到 {word2} 不可达")

    def _highlight_path_on_graph(self, path, dist, filename):
        """将最短路径标注在图上并展示"""
        try:
            os.environ["GRAPHVIZ_DOT"] = r"D://Graphviz-12.2.1-win64//bin//dot.exe"
            dot = Digraph(comment='Text Graph with Shortest Path', engine='dot')
            dot.attr(rankdir='LR', size='20,20')

            for node in self.nodes:
                dot.node(node, shape='circle')

            for from_node in self.graph:
                for to_node in self.graph[from_node]:
                    weight = self.graph[from_node][to_node]
                    if (from_node, to_node) in zip(path, path[1:]):
                        dot.edge(from_node, to_node, label=str(weight), fontsize='10', color='red', penwidth='3')
                    else:
                        dot.edge(from_node, to_node, label=str(weight), fontsize='10')

            dot.attr('node', shape='circle', fontname='Arial')
            dot.attr('edge', fontname='Arial', fontsize='10')

            filepath = dot.render(filename=filename, format='png', cleanup=True)
            print(f"标注最短路径的图已保存到 {filepath}")
            if sys.platform == 'win32':
                os.startfile(filepath)
            elif sys.platform == 'darwin':
                os.system(f'open "{filepath}"')
            else:
                os.system(f'xdg-open "{filepath}"')
        except Exception as e:
            print(f"生成图形时出错: {e}")

    def calculate_tf_idf(self, text):
        words = self.process_text(text)
        word_count = defaultdict(int)
        for word in words:
            word_count[word] += 1

        total_words = len(words)
        tf = {word: count / total_words for word, count in word_count.items()}

        document_count = 1  # 假设只有一个文档
        idf = {}
        for word in self.nodes:
            idf[word] = log(document_count / (1 + sum(1 for w in words if w == word)))

        tf_idf = {word: tf[word] * idf[word] for word in self.nodes}
        return tf_idf

    def calculate_pagerank(self, damping_factor=0.85, max_iterations=100, tolerance=1e-6, text=None):
        """计算PageRank值"""
        num_nodes = len(self.nodes)
        # 使用TF-IDF计算初始PR值
        if text:
            tf_idf = self.calculate_tf_idf(text)
            total_tf_idf = sum(tf_idf.values())
            pagerank = {node: tf_idf[node] / total_tf_idf for node in self.nodes}
        else:
            pagerank = {node: 1 / num_nodes for node in self.nodes}

        out_degree = {node: sum(self.graph[node].values()) for node in self.nodes}

        for _ in range(max_iterations):
            new_pagerank = {node: (1 - damping_factor) / num_nodes for node in self.nodes}
            # 先处理正常的节点
            for node in self.nodes:
                if out_degree[node] > 0:
                    for neighbor in self.graph[node]:
                        new_pagerank[neighbor] += damping_factor * pagerank[node] / out_degree[node]

            # 处理出度为0的节点
            zero_out_degree_nodes = [node for node in self.nodes if out_degree[node] == 0]
            zero_out_degree_pr = sum(pagerank[node] for node in zero_out_degree_nodes)
            for node in self.nodes:
                if out_degree[node] > 0:
                    new_pagerank[node] += damping_factor * zero_out_degree_pr / (num_nodes - len(zero_out_degree_nodes))

            # 检查收敛
            if sum(abs(new_pagerank[node] - pagerank[node]) for node in self.nodes) < tolerance:
                break

            pagerank = new_pagerank

        return pagerank


def command_line_main():
    parser = argparse.ArgumentParser(description="文本到有向图转换器")
    parser.add_argument('file', nargs='?', help="输入文本文件路径")
    parser.add_argument('--save-graph', metavar='FILENAME', help="保存有向图到文件(支持.png, .pdf, .svg)")
    args = parser.parse_args()

    if args.file:
        file_path = args.file
    else:
        file_path = input("请输入文本文件路径: ")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().replace('\n', ' ')
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到")
        sys.exit(1)

    graph = TextGraph()
    words = graph.process_text(text)
    graph.build_graph(words)

    if args.save_graph:
        saved_file = graph.save_graph_image(filename=args.save_graph)
        if saved_file:
            print(f"提示: 有向图已保存到 {saved_file}")
        else:
            print("错误: 保存有向图失败")
        return

    while True:
        print("\n请选择操作:")
        print("1. 显示图结构")
        print("2. 保存有向图图片")
        print("3. 查找桥接词")
        print("4. 生成新文本(插入桥接词)")
        print("5. 查找最短路径")
        print("6. 计算PageRank")
        print("7. 随机游走")
        print("8. 退出")

        choice = input("请输入选项(1-8): ")

        if choice == '1':
            graph.show_graph()
        elif choice == '2':
            filename = input("输入文件名(默认text_graph.png): ") or "text_graph.png"
            filepath, fileext = os.path.splitext(filename)
            format = fileext[1:] if fileext else 'png'
            saved_file = graph.save_graph_image(filename=filepath, format=format)
            if saved_file:
                print(f"提示: 有向图已保存到 {saved_file}")
            else:
                print("错误: 保存有向图失败")
        elif choice == '3':
            word1 = input("输入第一个单词: ").lower()
            word2 = input("输入第二个单词: ").lower()
            if not word1 or not word2:
                print("提示: 请输入两个单词")
                continue
            if word1 not in graph.nodes or word2 not in graph.nodes:
                print(f"No {word1} or {word2} in the graph!")
                continue
            bridges = graph.find_bridge_words(word1, word2)
            if bridges:
                if len(bridges) == 1:
                    print(f"The bridge word from {word1} to {word2} is: {bridges[0]}.")

                else:
                    last = bridges[-1]
                    others = ', '.join(bridges[:-1])
                    print(f"The bridge words from {word1} to {word2} are: {others}, and {last}.")
            else:
                print(f"No bridge words from {word1} to {word2}!")
        elif choice == '4':
            input_text = input("请输入新文本: ")
            new_text = graph.generate_new_text(input_text)
            print("\n生成的新文本:")
            print(new_text)
        elif choice == '5':
            words = input("请输入一个或两个单词（用空格分隔）: ").lower().split()
            if len(words) == 1:
                graph.show_shortest_path_on_graph(words[0])
            elif len(words) == 2:
                graph.show_shortest_path_on_graph(words[0], words[1])
            else:
                print("提示: 输入格式错误，请输入一个或两个单词")

        elif choice == '6':
            use_file_text = input("是否使用文件中的文本计算PageRank？(y/n): ").lower()
            if use_file_text == 'y':
                pagerank = graph.calculate_pagerank(text=text)
            else:
                custom_text = input("请输入自定义文本: ")
                pagerank = graph.calculate_pagerank(text=custom_text)

            sorted_pagerank = sorted(pagerank.items(), key=lambda item: item[1], reverse=True)
            print("\nPageRank值（按重要性排序）:")
            for node, score in sorted_pagerank:
                print(f"{node}: {score}")


        elif choice == '7':
            print("\n提示: 开始随机游走(遇到重复边或没有出边时停止，输入'stop'可提前终止)")
            path, total_weight, edges = graph.random_walk()
            print("\n提示: 随机游走完成!")
            print(f"最终路径: {' -> '.join(path)}")
            print(f"遍历边数: {len(edges)}, 总权重: {total_weight}")

            save = input("是否保存结果到文件?(y/n): ").lower()
            if save == 'y':
                filename = input("输入文件名(默认random_walk.txt): ") or "random_walk.txt"
                graph.save_walk_result(path, edges, filename)
        elif choice == '8':
            break
        else:
            print("提示: 无效选项，请重新输入")


if __name__ == "__main__":
    command_line_main()
    