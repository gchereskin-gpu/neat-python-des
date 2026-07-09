from neat.graphs import required_for_output
import math


class RecurrentNetwork:
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals

        self.values = [{}, {}]
        for v in self.values:
            for k in [*inputs, *outputs]:
                v[k] = 0.0

            for node, ignored_activation, ignored_aggregation, ignored_bias, ignored_response, links in self.node_evals:
                v[node] = 0.0
                for i, w in links:
                    v[i] = 0.0
        self.active = 0

    def reset(self):
        self.values = [{k: 0.0 for k in v} for v in self.values]
        self.active = 0

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError(f"Expected {len(self.input_nodes):n} inputs, got {len(inputs):n}")

        ivalues = self.values[self.active]
        ovalues = self.values[1 - self.active]
        self.active = 1 - self.active

        for i, v in zip(self.input_nodes, inputs):
            ivalues[i] = v
            ovalues[i] = v

        for node, activation, aggregation, bias, response, links in self.node_evals:
            node_inputs = [ivalues[i] * w for i, w in links]
            s = aggregation(node_inputs)
            ovalues[node] = activation(bias + response * s)

        return [ovalues[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a RecurrentNetwork). """
        genome_config = config.genome_config
        required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)

        # Gather inputs and expressed connections.
        node_inputs = {}
        for cg in genome.connections.values():
            if not cg.enabled:
                continue

            i, o = cg.key
            if o not in required and i not in required:
                continue

            if o not in node_inputs:
                node_inputs[o] = [(i, cg.weight)]
            else:
                node_inputs[o].append((i, cg.weight))

        node_evals = []
        for node_key, inputs in node_inputs.items():
            node = genome.nodes[node_key]
            activation_function = genome_config.activation_defs.get(node.activation)
            aggregation_function = genome_config.aggregation_function_defs.get(node.aggregation)
            node_evals.append((node_key, activation_function, aggregation_function, node.bias, node.response, inputs))

        return RecurrentNetwork(genome_config.input_keys, genome_config.output_keys, node_evals)

class AdaptiveESRecurrentNetwork:
    def __init__(self, inputs, outputs, node_evals, max_weight):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.max_weight = max_weight

        self.std_values = [{}, {}]
        for v in self.std_values:
            for k in [*inputs, *outputs]:
                v[k] = 0.0

            for node, ignored_activation, ignored_aggregation, ignored_bias, ignored_response, std_links, mod_links in self.node_evals:
                v[node] = 0.0
                if std_links:
                    for i, w, a, b, c, d, n in std_links:
                        v[i] = 0.0
        
        self.mod_values = [{}, {}]
        for v in self.mod_values:
            for k in [*inputs, *outputs]:
                v[k] = 0.0

            for node, ignored_activation, ignored_aggregation, ignored_bias, ignored_response, std_links, mod_links in self.node_evals:
                v[node] = 0.0
                if mod_links:
                    for i, w in mod_links:
                        v[i] = 0.0

        # store initial evolved weights for resetting the network.
        # Elements are immutable tuples, so a shallow list copy is enough;
        # None (mod-only nodes) is stored as-is.
        self.initial_weights = []
        for _, _, _, _, _, std_links, _ in self.node_evals:
            self.initial_weights.append(list(std_links) if std_links else std_links)

        self.active = 0

    def reset(self):
        self.std_values = [{k: 0.0 for k in v} for v in self.std_values]
        self.mod_values = [{k: 0.0 for k in v} for v in self.mod_values]
        self.active = 0
        # Restore each std_links list in place so the snapshot stays pristine
        # across resets and node_evals keeps the same list objects.
        for idx, (_, _, _, _, _, std_links, _) in enumerate(self.node_evals):
            if std_links:
                std_links[:] = self.initial_weights[idx]

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError(f"Expected {len(self.input_nodes):n} inputs, got {len(inputs):n}")

        std_ivalues = self.std_values[self.active]
        std_ovalues = self.std_values[1 - self.active]
        mod_ivalues = self.mod_values[self.active]
        mod_ovalues = self.mod_values[1 - self.active]
        self.active = 1 - self.active

        for i, v in zip(self.input_nodes, inputs):
            std_ivalues[i] = v
            std_ovalues[i] = v
            mod_ivalues[i] = v
            mod_ovalues[i] = v

        for node, activation, aggregation, bias, response, std_links, mod_links in self.node_evals:
            # update standard activations
            if std_links:
                node_inputs = [std_ivalues[i] * w for i, w, _, _, _, _, _ in std_links]
                s = aggregation(node_inputs)
                std_ovalues[node] = activation(bias + response * s)

            if mod_links:
                # update modulatory activations
                node_inputs = [mod_ivalues[i] * w for i, w in mod_links]
                s = aggregation(node_inputs)
                mod_ovalues[node] = activation(bias + response * s)

        # update weights based on modulatory activations and plasticity rules
        for node, activation, aggregation, bias, response, std_links, mod_links in self.node_evals:
            if std_links:
                # No incoming modulatory connections -> unmodulated (factor 1.0).
                # Has modulatory connections -> use their activation, even if it's 0.0.
                mod_factor = math.tanh(mod_ovalues[node] / 2) if mod_links else 1.0
                for idx, std_link in enumerate(std_links):
                    i, w, a, b, c, d, n = std_link
                    d_w = mod_factor * n * ((a * std_ovalues[i] * std_ovalues[node]) +
                                            (b * std_ovalues[i]) +
                                            (c * std_ovalues[node]) +
                                            (d * w))
                    new_w = max(-self.max_weight, min(w + d_w, self.max_weight))
                    std_links[idx] = (i, new_w, a, b, c, d, n)
        
        return [std_ovalues[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a RecurrentNetwork). """
        
        genome_config = config.genome_config
        required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)

        # Gather inputs and expressed connections.
        node_inputs = {}
        for cg in genome.connections.values():
            if not cg.enabled:
                continue

            i, o = cg.key
            if o not in required and i not in required:
                continue

            if o not in node_inputs:
                node_inputs[o] = [(i, cg.weight)]
            else:
                node_inputs[o].append((i, cg.weight))

        node_evals = []
        for node_key, inputs in node_inputs.items():
            node = genome.nodes[node_key]
            activation_function = genome_config.activation_defs.get(node.activation)
            aggregation_function = genome_config.aggregation_function_defs.get(node.aggregation)
            node_evals.append((node_key, activation_function, aggregation_function, node.bias, node.response, inputs))

        return AdaptiveESRecurrentNetwork(genome_config.input_keys, genome_config.output_keys, node_evals)

class AdaptiveRecurrentNetwork:
    def __init__(self, inputs, outputs, node_evals, max_weight, num_branches):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.max_weight = max_weight
        self.num_branches = num_branches

        self.values = [{}, {}]
        for v in self.values:
            for k in [*inputs, *outputs]:
                v[k] = 0.0

            for node, ignored_activation, ignored_aggregation, ignored_bias, ignored_response, links in self.node_evals:
                v[node] = 0.0
                for i, w, b_id, a, b, c, d, n, mod_w in links:
                    v[i] = 0.0

        # store initial evolved weights for resetting the network.
        # Elements are immutable tuples (mod_w dicts are shared but never
        # mutated, only w changes), so a shallow list copy is enough.
        self.initial_weights = [list(links) for _, _, _, _, _, links in self.node_evals]

        self.active = 0

    def reset(self):
        self.values = [{k: 0.0 for k in v} for v in self.values]
        self.active = 0
        # Restore each links list in place so the snapshot stays pristine
        # across resets and node_evals keeps the same list objects.
        for idx, (_, _, _, _, _, links) in enumerate(self.node_evals):
            links[:] = self.initial_weights[idx]

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError(f"Expected {len(self.input_nodes):n} inputs, got {len(inputs):n}")

        ivalues = self.values[self.active]
        ovalues = self.values[1 - self.active]
        self.active = 1 - self.active

        for i, v in zip(self.input_nodes, inputs):
            ivalues[i] = v
            ovalues[i] = v

        for node, activation, aggregation, bias, response, links in self.node_evals:
            node_inputs = [ivalues[i] * w for i, w, b_id, a, b, c, d, n, mod_w in links]
            s = aggregation(node_inputs)
            ovalues[node] = activation(bias + response * s)
        
        # compute modulatory activations
        mod_activations = {node:{} for node in ovalues.keys()}
        for branch_id in range(self.num_branches):
            for node, activation, aggregation, bias, response, links in self.node_evals:
                node_inputs = []
                for i, w, b_id, a, b, c, d, n, mod_w in links:
                    if mod_w.get(branch_id):
                        node_inputs.append(ovalues[i] * mod_w[branch_id])
                s = aggregation(node_inputs)
                mod_activations[node][branch_id] = activation(bias + response * s)

        # update connection weights with learning rules and modulatory activations
        # for each connection, its plasticity is modulated by the postsynaptic node's modulatory activation - specifically the activation for the connection's branch
        for node, activation, aggregation, bias, response, links in self.node_evals:
            for idx, link in enumerate(links):
                i, w, bid, a, b, c, d, n, mod_w = link
                d_w = math.tahn(mod_activations[node].get(bid, 1.0)/2) * n * ((a * ovalues[i] * ovalues[node]) +
                                                                (b * ovalues[i]) +
                                                                (c * ovalues[node]) +
                                                                (d * w))
                new_w = max(-self.max_weight, min(w + d_w, self.max_weight))
                links[idx] = (i, new_w, bid, a, b, c, d, n, mod_w)

        return [ovalues[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a RecurrentNetwork). """
        #TODO: Needs updating if this method is ever used

        genome_config = config.genome_config
        required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)

        # Gather inputs and expressed connections.
        node_inputs = {}
        for cg in genome.connections.values():
            if not cg.enabled:
                continue

            i, o = cg.key
            if o not in required and i not in required:
                continue

            if o not in node_inputs:
                node_inputs[o] = [(i, cg.weight)]
            else:
                node_inputs[o].append((i, cg.weight))

        node_evals = []
        for node_key, inputs in node_inputs.items():
            node = genome.nodes[node_key]
            activation_function = genome_config.activation_defs.get(node.activation)
            aggregation_function = genome_config.aggregation_function_defs.get(node.aggregation)
            node_evals.append((node_key, activation_function, aggregation_function, node.bias, node.response, inputs))

        return AdaptiveRecurrentNetwork(genome_config.input_keys, genome_config.output_keys, node_evals)