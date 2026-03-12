from agent import build_sales_agent


def run_cli() -> None:
    agent_executor = build_sales_agent(verbose=True)
    print("Sales Assistant Agent ready. Type your question, or 'exit' to quit.")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not question:
            continue

        response = agent_executor.invoke({"input": question})
        print(f"Agent: {response.get('output', 'No answer generated.')}\n")


if __name__ == "__main__":
    run_cli()
