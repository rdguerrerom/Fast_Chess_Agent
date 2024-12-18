# Import necessary libraries
from kaggle_environments import make

# Create the environment
env = make("chess", debug=True)

# Initialize counters
agent_wins = 0
agent_wins_as_white = 0
agent_wins_as_black = 0
opponent_wins = 0
draws = 0
match_count = 0

# Flag to alternate agent's play as white or black
agent_plays_white = True

# Loop until the agent has 15 wins
while agent_wins < 5:
    match_count += 1
    print(f"Match {match_count} in progress...")

    # Alternate agent's side
    if agent_plays_white:
        result = env.run(["submission.py", "random"])
    else:
        result = env.run(["random", "submission.py"])

    try:
        # Extract rewards
        agent_reward = result[-1][0].reward if agent_plays_white else result[-1][1].reward
        opponent_reward = result[-1][1].reward if agent_plays_white else result[-1][0].reward

        # Replace None rewards with 0
        agent_reward = agent_reward if agent_reward is not None else 0
        opponent_reward = opponent_reward if opponent_reward is not None else 0

        # Get match statuses
        agent_status = result[-1][0].status if agent_plays_white else result[-1][1].status
        opponent_status = result[-1][1].status if agent_plays_white else result[-1][0].status

        # Determine match outcome
        side_played = "White" if agent_plays_white else "Black"

        if agent_status == "DONE" and agent_reward > opponent_reward:
            # Agent won
            agent_wins += 1
            if agent_plays_white:
                agent_wins_as_white += 1
            else:
                agent_wins_as_black += 1

            print(f"Agent Win #{agent_wins} playing as {side_played}!")

            # Render the winning match
            print(f"Rendering Agent Win #{agent_wins}...")
            env.render(mode="ipython", width=1000, height=1000)

        elif opponent_status == "DONE" and opponent_reward > agent_reward:
            opponent_wins += 1
            print("Opponent wins!")
        else:
            draws += 1
            print("Match ended in a draw")

        # Partial summary report
        print(f"\nPartial Results after {match_count} Matches:")
        print(f"Agent Wins: {agent_wins} (White: {agent_wins_as_white}, Black: {agent_wins_as_black})")
        print(f"Opponent Wins: {opponent_wins}")
        print(f"Draws: {draws}")
        print("-" * 30)

    except Exception as e:
        print(f"Error processing match result: {e}")
        draws += 1  # Conservatively count as a draw if processing fails

    # Toggle agent's side
    agent_plays_white = not agent_plays_white

    # Safety check to prevent infinite loop
    if match_count >= 250:
        print("Maximum match limit reached")
        break

# Final report
print("\nFinal Results:")
print(f"Total Matches Played: {match_count}")
print(f"Agent Wins: {agent_wins} (White: {agent_wins_as_white}, Black: {agent_wins_as_black})")
print(f"Opponent Wins: {opponent_wins}")
print(f"Draws: {draws}")

