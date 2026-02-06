from research_agent_crew.crew import ResearchAgentCrew


def run():
    """
    Run the crew.
    """
    inputs = {
        'query': 'What is incontext learning?',
    }

    try:
        ResearchAgentCrew().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
