# -*- coding: utf-8 -*-

import csv
import json
import threading
import time
from argparse import ArgumentParser

import requests
from flask import Flask, jsonify, request

class Router:
    """
    Representa um roteador que executa o algoritmo de Vetor de Distância.
    """

    def __init__(self, my_address, neighbors, my_network, update_interval=1):
        """
        Inicializa o roteador.

        :param my_address: O endereço (ip:porta) deste roteador.
        :param neighbors: Um dicionário contendo os vizinhos diretos e o custo do link.
                          Ex: {'127.0.0.1:5001': 5, '127.0.0.1:5002': 10}
        :param my_network: A rede que este roteador administra diretamente.
                           Ex: '10.0.1.0/24'
        :param update_interval: O intervalo em segundos para enviar atualizações, o tempo que o roteador espera 
                                antes de enviar atualizações para os vizinhos.        """
        self.my_address = my_address
        self.neighbors = neighbors
        self.my_network = my_network
        self.update_interval = update_interval

        # TODO: Este é o local para criar e inicializar sua tabela de roteamento.
        #
        # 1. Crie a estrutura de dados para a tabela de roteamento. Um dicionário é
        #    uma ótima escolha, onde as chaves são as redes de destino (ex: '10.0.1.0/24')
        #    e os valores são outro dicionário contendo 'cost' e 'next_hop'.
        #    Ex: {'10.0.1.0/24': {'cost': 0, 'next_hop': '10.0.1.0/24'}}
        self.routing_table = {}
        #
        # 2. Adicione a rota para a rede que este roteador administra diretamente
        #    (a rede em 'self.my_network'). O custo para uma rede diretamente
        #    conectada é 0, e o 'next_hop' pode ser a própria rede ou o endereço do roteador.
        #
        self.routing_table[self.my_network] = {
            'cost': 0,
            'next_hop': self.my_network
        }
        # 3. Adicione as rotas para seus vizinhos diretos, usando o dicionário
        #    'self.neighbors'. Para cada vizinho, o 'cost' é o custo do link direto
        #    e o 'next_hop' é o endereço do próprio vizinho.
        for neighbor_address, cost in self.neighbors.items():
                # A chave da tabela é o endereço do vizinho, que funciona como o destino inicial.
                # O roteador aprenderá as *redes* dos vizinhos quando receber atualizações deles.
                self.routing_table[neighbor_address] = {
                    'cost': cost,
                    'next_hop': neighbor_address
                }

        print("Tabela de roteamento inicial:")
        print(json.dumps(self.routing_table, indent=4))

        # Inicia o processo de atualização periódica em uma thread separada
        self._start_periodic_updates()

    def _start_periodic_updates(self):
        """Inicia uma thread para enviar atualizações periodicamente."""
        thread = threading.Thread(target=self._periodic_update_loop)
        thread.daemon = True
        thread.start()

    def _periodic_update_loop(self):
        """Loop que envia atualizações de roteamento em intervalos regulares."""
        while True:
            time.sleep(self.update_interval)
            print(f"[{time.ctime()}] Enviando atualizações periódicas para os vizinhos...")
            try:
                self.send_updates_to_neighbors()
            except Exception as e:
                print(f"Erro durante a atualização periódida: {e}")

    def _get_summarized_table(self):
        """
        Cria uma cópia da tabela de roteamento e aplica a sumarização de rotas.
        Retorna a tabela sumarizada.
        """
        # 1. Crie uma cópia profunda para não modificar a tabela original.
        summarized_table = copy.deepcopy(self.routing_table)

        # Agrupa rotas pelo mesmo 'next_hop', pois é uma condição para sumarizar.
        rotas_por_hop = {}
        for network, info in self.routing_table.items():
            next_hop = info['next_hop']
            if next_hop not in rotas_por_hop:
                rotas_por_hop[next_hop] = []
            rotas_por_hop[next_hop].append(network)

        # Itera sobre cada grupo de rotas com o mesmo next_hop.
        for next_hop, networks in rotas_por_hop.items():
            houve_sumarizacao = True
            # Continua tentando sumarizar enquanto for possível.
            while houve_sumarizacao:
                houve_sumarizacao = False
                # Usamos combinations para pegar todos os pares de redes.
                from itertools import combinations
                
                # Cópia da lista para poder modificar durante a iteração
                for net1, net2 in combinations(list(networks), 2):
                    nova_rede = tentar_sumarizar(net1, net2)
                    
                    if nova_rede:
                        # Sumarização bem-sucedida!
                        print(f"Sumarizando {net1} e {net2} -> {nova_rede}")
                        
                        # Remove as rotas específicas da tabela sumarizada.
                        info1 = summarized_table.pop(net1)
                        info2 = summarized_table.pop(net2)
                        
                        # Adiciona a nova rota sumarizada.
                        # O custo é o MAIOR entre as rotas originais.
                        summarized_table[nova_rede] = {
                            'cost': max(info1['cost'], info2['cost']),
                            'next_hop': next_hop
                        }
                        
                        # Atualiza a lista de redes para a próxima iteração.
                        networks.remove(net1)
                        networks.remove(net2)
                        networks.append(nova_rede)
                        
                        houve_sumarizacao = True
                        # Quebra o loop for para reiniciar a busca por pares na lista atualizada.
                        break 
        
        return summarized_table


    def send_updates_to_neighbors(self):
        """
        Envia a tabela de roteamento (potencialmente sumarizada) para todos os vizinhos.
        """
        # TODO: O código abaixo envia a tabela de roteamento *diretamente*.
        #
        # ESTE TRECHO DEVE SER CHAMAADO APOS A SUMARIZAÇÃO.
        #
        # dica:
        # 1. CRIE UMA CÓPIA da `self.routing_table` NÃO ALTERE ESTA VALOR.
        # 2. IMPLEMENTE A LÓGICA DE SUMARIZAÇÃO nesta cópia.
        print("Gerando tabela sumarizada...")
        tabela_para_enviar = self._get_summarized_table()
        print("Tabela final a ser enviada:")
        print(json.dumps(tabela_para_enviar, indent=4))
        # 3. ENVIE A CÓPIA SUMARIZADA no payload, em vez da tabela original.
        
        payload = {
            "sender_address": self.my_address,
            "routing_table": tabela_para_enviar
        }

        for neighbor_address in self.neighbors:
            url = f'http://{neighbor_address}/receive_update'
            try:
                print(f"Enviando tabela para {neighbor_address}")
                requests.post(url, json=payload, timeout=5)
            except requests.exceptions.RequestException as e:
                print(f"Não foi possível conectar ao vizinho {neighbor_address}. Erro: {e}")

# --- API Endpoints ---
# Instância do Flask e do Roteador (serão inicializadas no main)
app = Flask(__name__)
router_instance = None

@app.route('/routes', methods=['GET'])
def get_routes():
    """Endpoint para visualizar a tabela de roteamento atual."""
    # TODO: Aluno! Este endpoint está parcialmente implementado para ajudar na depuração.
    # Você pode mantê-lo como está ou customizá-lo se desejar.
    # - mantenha o routing_table como parte da resposta JSON.
    if router_instance:
        # --- INÍCIO DA IMPLEMENTAÇÃO DO PASSO 4 ---

        # Simplesmente retornamos um JSON com as informações atuais do roteador.
        # A chave "routing_table" é a mais importante.
        return jsonify({
            "my_address": router_instance.my_address,
            "my_network": router_instance.my_network,
            "neighbors" : router_instance.neighbors,
            "routing_table": router_instance.routing_table
        })

        # --- FIM DA IMPLEMENTAÇÃO DO PASSO 4 ---
    
    return jsonify({"error": "Roteador não inicializado"}), 500

@app.route('/receive_update', methods=['POST'])
def receive_update():
    """Endpoint que recebe atualizações de roteamento de um vizinho."""
    if not request.json:
        return jsonify({"error": "Invalid request"}), 400

    update_data = request.json
    sender_address = update_data.get("sender_address")
    sender_table = update_data.get("routing_table")

    if not sender_address or not isinstance(sender_table, dict):
        return jsonify({"error": "Missing sender_address or routing_table"}), 400

    print(f"Recebida atualização de {sender_address}:")
    print(json.dumps(sender_table, indent=4))

    # TODO: Implemente a lógica de Bellman-Ford aqui.
    table_changed = False
    #
    # 1. Verifique se o remetente é um vizinho conhecido.
    if sender_address not in router_instance.neighbors:
        print(f"Aviso: Recebida atualização de um não-vizinho {sender_address}. Ignorando.")
        # Retorna sucesso, pois não é um erro, apenas uma atualização a ser ignorada.
        return jsonify({"status": "ignored", "message": "Sender is not a known neighbor"}), 200
    # 2. Obtenha o custo do link direto para este vizinho a partir de `router_instance.neighbors`.
    cost_to_neighbor = router_instance.neighbors[sender_address]
    # 3. Itere sobre cada rota (`network`, `info`) na `sender_table` recebida.
    for network, info in sender_table.items():
        # Prevenção de loop simples (Split Horizon): não considere rotas que o vizinho
        # aprendeu através de você.
        if info['next_hop'] == router_instance.my_address:
            continue

        new_cost = cost_to_neighbor + info['cost']

        # 5. Verifique sua própria tabela de roteamento:
        #    a. Se você não conhece a `network`...
        if network not in router_instance.routing_table:
            router_instance.routing_table[network] = {
                'cost': new_cost,
                'next_hop': sender_address
            }
            table_changed = True
        else:
            # Se você já conhece a `network`...
            current_route_info = router_instance.routing_table[network]
            current_cost = current_route_info['cost']
            current_next_hop = current_route_info['next_hop']

            #    b. Se o novo custo é menor que o custo atual...
            is_cheaper_path = new_cost < current_cost
            
            #    c. Ou se a rota atual passa pelo remetente de qualquer maneira...
            #       (Isso é crucial para propagar aumentos de custo, ex: link ficou mais lento)
            is_path_via_sender = current_next_hop == sender_address

            if is_cheaper_path or is_path_via_sender:
                # Apenas atualize se o custo realmente mudou para evitar logs desnecessários
                if current_cost != new_cost:
                    current_route_info['cost'] = new_cost
                    current_route_info['next_hop'] = sender_address
                    table_changed = True

    # 6. Se a tabela mudou, imprima a nova tabela no console.
    if table_changed:
        print("\n--- Tabela de Roteamento ATUALIZADA ---")
        print(json.dumps(router_instance.routing_table, indent=4))
        print("--------------------------------------\n")
  

    return jsonify({"status": "success", "message": "Update received"}), 200

if __name__ == '__main__':
    parser = ArgumentParser(description="Simulador de Roteador com Vetor de Distância")
    parser.add_argument('-p', '--port', type=int, default=5000, help="Porta para executar o roteador.")
    parser.add_argument('-f', '--file', type=str, required=True, help="Arquivo CSV de configuração de vizinhos.")
    parser.add_argument('--network', type=str, required=True, help="Rede administrada por este roteador (ex: 10.0.1.0/24).")
    parser.add_argument('--interval', type=int, default=10, help="Intervalo de atualização periódica em segundos.")
    args = parser.parse_args()

    # Leitura do arquivo de configuração de vizinhos
    neighbors_config = {}
    try:
        with open(args.file, mode='r') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                neighbors_config[row['vizinho']] = int(row['custo'])
    except FileNotFoundError:
        print(f"Erro: Arquivo de configuração '{args.file}' não encontrado.")
        exit(1)
    except (KeyError, ValueError) as e:
        print(f"Erro no formato do arquivo CSV: {e}. Verifique as colunas 'vizinho' e 'custo'.")
        exit(1)

    my_full_address = f"127.0.0.1:{args.port}"
    print("--- Iniciando Roteador ---")
    print(f"Endereço: {my_full_address}")
    print(f"Rede Local: {args.network}")
    print(f"Vizinhos Diretos: {neighbors_config}")
    print(f"Intervalo de Atualização: {args.interval}s")
    print("--------------------------")

    router_instance = Router(
        my_address=my_full_address,
        neighbors=neighbors_config,
        my_network=args.network,
        update_interval=args.interval
    )

    # Inicia o servidor Flask
    app.run(host='0.0.0.0', port=args.port, debug=False)

# --- FUNÇÕES AUXILIARES PARA MANIPULAÇÃO DE REDES (PASSO 3) ---

def ip_para_int(ip_str):
    """Converte um endereço IP string (ex: '192.168.1.0') para um inteiro de 32 bits."""
    octetos = ip_str.split('.')
    return (int(octetos[0]) << 24) + (int(octetos[1]) << 16) + (int(octetos[2]) << 8) + int(octetos[3])

def int_para_ip(ip_int):
    """Converte um inteiro de 32 bits de volta para um endereço IP string."""
    return f"{(ip_int >> 24) & 0xFF}.{(ip_int >> 16) & 0xFF}.{(ip_int >> 8) & 0xFF}.{ip_int & 0xFF}"

def tentar_sumarizar(net1_str, net2_str):
    """
    Tenta sumarizar duas redes no formato CIDR (ex: '192.168.20.0/24').
    Retorna a nova rede sumarizada se for possível, caso contrário, retorna None.
    """
    try:
        ip1_str, pref1_str = net1_str.split('/')
        ip2_str, pref2_str = net2_str.split('/')
        prefixo1, prefixo2 = int(pref1_str), int(pref2_str)

        # Condição 1: As redes devem ter o mesmo tamanho de prefixo para serem agregadas.
        if prefixo1 != prefixo2:
            return None

        # O novo prefixo será 1 bit menor (bloco maior).
        novo_prefixo = prefixo1 - 1
        if novo_prefixo < 0: # Não pode ser menor que 0
            return None

        # Converte IPs para inteiros para a mágica de bits.
        ip1_int = ip_para_int(ip1_str)
        ip2_int = ip_para_int(ip2_str)

        # Cria a máscara para a super-rede. Ex: para /23, a máscara tem 23 bits '1'.
        # `(1 << (32 - novo_prefixo)) - 1` cria os bits '0' à direita.
        # `~` inverte, criando os bits '1' à esquerda (a máscara).
        mascara = ~((1 << (32 - novo_prefixo)) - 1) & 0xFFFFFFFF

        # Condição 2: Ambas as redes originais, quando mascaradas com a nova
        # máscara da super-rede, devem resultar na mesma rede base.
        if (ip1_int & mascara) == (ip2_int & mascara):
            # Sucesso! A rede base é o resultado da operação AND.
            nova_rede_int = ip1_int & mascara
            return f"{int_para_ip(nova_rede_int)}/{novo_prefixo}"
        else:
            # As redes não são "vizinhas" contíguas para esta agregação.
            return None
    except (ValueError, IndexError):
        # Trata casos onde o formato da string não é 'ip/prefixo'.
        return None