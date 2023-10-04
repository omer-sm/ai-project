import requests
import json

server_address = "http://localhost:8080"

def update_cost_and_iteration(cost, iteration):
    cost_data = {"cost": cost}
    requests.post(server_address + "/costs", data=json.dumps(cost_data))
    iteration_data = {"iteration": iteration}
    requests.post(server_address + "/stats", data=json.dumps(iteration_data))


def update_initial_stats(iterations, iterations_per_send):
    requests.post(server_address + "/reset", data={})
    stats_data = {"totalIterations": iterations, "iterationsPerSend": iterations_per_send}
    requests.post(server_address + "/stats", data=json.dumps(stats_data))