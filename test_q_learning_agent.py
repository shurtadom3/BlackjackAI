from src.agents.q_learning_agent import QLearningAgent

agent = QLearningAgent()

print("Agent created successfully.")
print("Initial epsilon:", agent.epsilon)
print("Initial Q-table size:", agent.get_q_table_size())

state = (15, 7, False)

action = agent.choose_action(state)

print("Selected action:", action)

assert action in [0, 1]

state = (15, 7, False)
action = 1
reward = -1
next_state = (20, 7, False)
done = True

print("Before update:", agent.q_table.get(state))

agent.update(
    state=state,
    action=action,
    reward=reward,
    next_state=next_state,
    done=done
)

print("After update:", agent.q_table[state])

print("Before decay:", agent.epsilon)

agent.decay_epsilon()

print("After decay:", agent.epsilon)
