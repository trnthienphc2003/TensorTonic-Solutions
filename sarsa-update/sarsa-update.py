def sarsa_update(q_table: list, state: int, action: int, reward: float, next_state: int, next_action: int, alpha: float, gamma: float) -> list:
    """
    Returns a copied Q-table after one SARSA update.
    """
    # Write code here
    q_new = q_table[:]
    delta = reward + gamma * q_new[next_state][next_action] - q_table[state][action]

    q_new[state][action] += alpha * delta
    return q_new