def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here

    n = len(values)
    m = len(transitions[0])
    ans = [float("-inf") for _ in range(n)]
    for s_next in range(n):
        for a in range(m):
            sum = 0
            for s in range(n):
                sum += transitions[s_next][a][s] * values[s]
            ans[s_next] = max(ans[s_next], rewards[s_next][a] + gamma * sum)
    return ans
                