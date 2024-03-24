import pandas as pd
import networkx as nx
import random
import matplotlib.pyplot as plt
from typing import Dict, List

global modularity
modularity_values = []


def load_graph(path: str) -> Dict[int, List[int]]:
    graph = {}
    data = pd.read_csv(path)

    for _, row in data.iterrows():
        source = int(row['source'])
        target = int(row['target'])

        if source in graph:
            graph[source].append(target)
        else:
            graph[source] = [target]

        if target in graph:
            graph[target].append(source)
        else:
            graph[target] = [source]

    return graph


# print the current situation of the graph per iteration
def print_current_graph(graph: Dict[int, List[int]], labels: Dict[int, int], iteration: int):
    """
    Print the graph with community labels for the specified iteration.
    """
    print(f"Iteration {iteration} - Graph with Community Labels:")
    draw_graph_with_communities(graph, labels)


def check_convergence(change_percentage: float, iteration: int, max_iterations: int,
                      convergence_threshold: float) -> bool:
    if change_percentage >= convergence_threshold:
        print("Convergence reached (change percentage)")
        return True
    elif iteration == max_iterations:
        print("Convergence reached (maximum iterations)")
        return True
    else:
        return False


def label_propagation(graph: Dict[int, List[int]], max_iterations: int = None,
                      convergence_threshold: float = None, freezing_threshold: float = None) -> Dict[int, int]:
    global modularity_values
    labels = {node: node for node in graph}  # Initialize labels
    modularity_values = []  # List to store modularity values for each iteration
    num_communities = []  # List to store the number of communities for each iteration
    total_changed_labels = 0
    flag_freeze = True

    # Calculate default values for max_iterations and convergence_threshold
    if max_iterations is None:
        max_iterations = int(len(graph) * 1 / 4)
        # max_iterations = int(len(graph))
    if convergence_threshold is None:
        convergence_threshold = 3.1
    if freezing_threshold is None:
        flag_freeze = False

    # counter to track on the change label's percentage
    # print_current_graph(graph, labels, 0)

    for iteration in range(1, max_iterations + 1):
        # copy the last iteration labels status.
        prev_labels = labels.copy()

        # not sure that it is needed:
        # # Shuffle nodes to break ties randomly
        nodes = list(graph.keys())
        # random.shuffle(nodes)
        iteration_changed_labels = 0

        for node in nodes:
            # create a list of node's neighbors labels from previous iteration
            neighbor_labels = [prev_labels[neighbor] for neighbor in graph[node]]
            # adding the node's label
            neighbor_labels.append(prev_labels[node])  # Include prior label

            # Count occurrences of each label
            label_counts = {label: neighbor_labels.count(label) for label in set(neighbor_labels)}

            # Determine the most frequent label (with random tie-breaking)
            max_count = max(label_counts.values())
            most_frequent_labels = [label for label, count in label_counts.items() if count == max_count]
            new_label = random.choice(most_frequent_labels)

            # Update label if it has changed
            if new_label != labels[node]:
                labels[node] = new_label
                total_changed_labels += 1
                iteration_changed_labels += 1

            # case the algorithm use the freezing mechanism
            if flag_freeze:
                iteration_change_percentage = iteration_changed_labels / len(labels)
                if iteration_change_percentage >= freezing_threshold:
                    break

        # Calculate percentage of labels changed
        change_percentage = total_changed_labels / len(labels)

        # Calculate modularity
        modularity = calculate_modularity(graph, labels)

        # Calculate the number of communities , by removing duplication with set.
        num_communities_iter = len(set(labels.values()))

        # Print iteration details
        print(f"Iteration {iteration}: Modularity = {modularity}, Label Changes Percentage = {change_percentage}, "
              f"Number of Communities = {num_communities_iter}")

        # Append modularity and number of communities to lists
        modularity_values.append(modularity)
        num_communities.append(num_communities_iter)

        # print_current_graph(graph, labels, iteration)

        # Check convergence criteria
        if check_convergence(change_percentage, iteration, max_iterations, convergence_threshold):
            break

    return labels


# def calculate_modularity(graph: Dict[int, List[int]], labels: Dict[int, int]) -> float:

#     G = nx.Graph(graph)
#     return nx.algorithms.community.modularity(G, [set(labels.keys())])
def calculate_modularity(graph: Dict[int, List[int]], labels: Dict[int, int]) -> float:
    G = nx.Graph(graph)

    # Convert labels dictionary into a partition format (list of sets)
    partition = {}
    for node, label in labels.items():
        if label in partition:
            partition[label].add(node)
        else:
            partition[label] = {node}

    # Calculate modularity using the partition
    return nx.algorithms.community.modularity(G, [nodes for nodes in partition.values()])


def draw_graph_with_labels(graph: Dict[int, List[int]]):
    G = nx.Graph(graph)
    pos = nx.spring_layout(G, seed=42)  # Positions for all nodes using the spring layout with a seed

    # Draw nodes with labels
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='skyblue', edgecolors='black', linewidths=2,
                           node_shape='o')
    # nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    plt.show()


def draw_graph_with_communities(graph: Dict[int, List[int]], labels: Dict[int, int]):
    G = nx.Graph(graph)
    pos = nx.spring_layout(G, seed=42)  # Positions for all nodes using the spring layout with a seed

    # Extract unique labels and assign a color to each label
    unique_labels = set(labels.values())
    color_map = {label: f'C{i}' for i, label in enumerate(unique_labels)}

    # Draw nodes with labels and colors
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color=[color_map[labels[node]] for node in G.nodes()],
                           edgecolors='black', linewidths=2, node_shape='o')
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # Show plot
    plt.show()


def print_modularity_per_iteration(modularity_vals: List[float]):
    """
    Print the modularity values for each iteration.
    """
    for i, modularity in enumerate(modularity_values, start=1):
        print(f"Iteration {i}: Modularity = {modularity}")


# Example usage:
# print_modularity_per_iteration(modularity_values)

def choose_graph():
    print("Hello Which graph to use?")
    print("1. Karate")
    print("2. Trump")
    print("3. Wikipedia vote")

    choice = int(input("Enter your choice: "))
    while 1:
        if choice == 1:
            return r"/Users/Metar/Documents/Social Network Analysis/Final Project/data sets/Karate.csv"
        elif choice == 2:
            return r"/Users/Metar/Documents/Social Network Analysis/Final Project/data sets/trump_new.csv"
        elif choice == 3:
            return r"/Users/Metar/Documents/Social Network Analysis/Final Project/data sets/Wikipedia vote.csv"
        else:
            print("Invalid choice. Please enter a valid choice (1-3).")
            print("1. Karate")
            print("2. Trump")
            print("3. Wikipedia vote")
            choice = int(input("Enter your choice: "))


def load_freezing_threshold():
    return float(input("please choose a freezing threshold: "))


def load_max_iteration_threshold():
    return int(input("please choose a max-iteration threshold: "))


def load_convergence_threshold():
    return float(input("please choose a max-iteration threshold: "))


def print_communities(labels: Dict[int, int]):
    # Group nodes by their labels to form communities
    communities = {}
    for node, label in labels.items():
        if label in communities:
            communities[label].append(node)
        else:
            communities[label] = [node]

    # Print communities and their nodes
    for label, nodes in communities.items():
        print(f"Community {label}:")
        print("Nodes:", nodes)


def show_menu():
    print("\nOptions:")
    print("1 - run algorithm")
    print("2 - print modularity per iteration")
    print("3 - change freezing threshold")
    print("4 - cancel freezing threshold")
    print("5 - change max-iteration threshold")
    print("6 - change convergence threshold")
    print("7 - change graph")
    print("8 - print communities")
    print("9 - print graph view")
    print("0 - Exit")

    return input("Enter an option (0-7): ")


def main():
    freezing_threshold = None
    max_iteration_threshold = None
    convergence_threshold = None
    labels = None
    file_path = choose_graph()
    choice = input("Do you want to add freezing mechanism? [y/n]: ")
    if choice == "y":
        freezing_threshold = load_freezing_threshold()
    graph = load_graph(file_path)
    choice = input("Do you want to use default thresholds ? [y/n]: ")
    if choice == "n":
        choice = input("Do you want to add max-iteration threshold? [y/n]: ")
        if choice == "y":
            max_iteration_threshold = load_max_iteration_threshold()
        choice = input("Do you want to add change's label percentage threshold? [y/n]: ")
        if choice == "y":
            convergence_threshold = load_convergence_threshold()

    while True:

        user_input = show_menu()

        if user_input == '0':
            break
        elif user_input == '1':
            labels = label_propagation(graph, max_iteration_threshold, convergence_threshold, freezing_threshold)
        elif user_input == '2':
            print_modularity_per_iteration(modularity_values)
        elif user_input == '3':
            freezing_threshold = load_freezing_threshold()
            print("Make sure you run the algorithm again")
        elif user_input == '4':
            freezing_threshold = None
            print("Make sure you run the algorithm again")
        elif user_input == '5':
            max_iteration_threshold = load_max_iteration_threshold()
            print("Make sure you run the algorithm again")
        elif user_input == '6':
            convergence_threshold = load_convergence_threshold()
            print("Make sure you run the algorithm again")
        elif user_input == '7':
            file_path = choose_graph()
            print("Make sure you run the algorithm again")
        elif user_input == '8':
            if labels is None:
                print("Please run the algorithm first")
            else:
                print_communities(labels)
        elif user_input == '9':
            if labels is None:
                draw_graph_with_labels(graph)
            else:
                draw_graph_with_communities(graph, labels)

        else:
            print("Invalid option. Please enter a number between 0 and 7.")


if __name__ == "__main__":
    main()
