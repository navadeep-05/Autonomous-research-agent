from agent.state import create_initial_state
from agent.planner import create_research_plan
from agent.tools.web_search import search_web
from agent.evidence import extract_evidence
from agent.source_evaluator import evaluate_source
from agent.gap_detector import detect_gaps
from agent.citation_tracker import create_source_records
from agent.claim_verifier import verify_claims
from agent.report_generator import generate_report


def run_research_agent(question):
    """
    Run the autonomous research process.
    """

    # Create the initial state
    state = create_initial_state(question)

    # -------------------------------
    # STEP 1: Create research plan
    # -------------------------------

    print("\nCreating research plan...")

    state["plan"] = create_research_plan(question)

    print("\n========== RESEARCH PLAN ==========\n")

    for index, task in enumerate(state["plan"], start=1):
        print(f"{index}. {task}")

    # -------------------------------
    # STEP 2: Initial research
    # -------------------------------

    for index, task in enumerate(state["plan"]):

        print(f"\nSearching task {index + 1}: {task}")

        state["current_task"] = index

        # Search the web
        result = search_web(task)

        # Store raw search result
        state["search_results"].append({
            "task": task,
            "result": result
        })

        # Extract useful evidence
        evidence = extract_evidence(task, result)

        # Store the evidence once
        state["evidence"].extend(evidence)

        # Evaluate the quality of the source
        evaluation = evaluate_source(result)

        # Store the evaluation
        state["source_evaluations"].append({
            "task": task,
            "evaluation": evaluation
        })

        # Count this research cycle
        state["iteration"] += 1

    # -------------------------------
    # STEP 3: Detect research gaps
    # -------------------------------

    print("\nDetecting research gaps...")

    state["gaps"] = detect_gaps(
        question,
        state["evidence"]
    )

    # -------------------------------
    # STEP 4: Re-search missing areas
    # -------------------------------

    max_research_rounds = 2

    for round_number in range(max_research_rounds):

        # Stop if there are no gaps
        if not state["gaps"]:
            print("\nNo major research gaps detected.")
            break

        print(
            f"\n========== RESEARCH ROUND "
            f"{round_number + 2} =========="
        )

        # Copy current gaps
        current_gaps = state["gaps"]

        # Clear old gaps before new research
        state["gaps"] = []

        # Research every identified gap
        for gap in current_gaps:

            print(f"\nSearching missing area: {gap}")

            # Search specifically for the missing information
            result = search_web(gap)

            # Store additional search result
            state["search_results"].append({
                "task": gap,
                "result": result
            })

            # Extract evidence from the new result
            evidence = extract_evidence(
                gap,
                result
            )

            # Store new evidence once
            state["evidence"].extend(evidence)

            # Evaluate the new source
            evaluation = evaluate_source(result)

            # Store source evaluation
            state["source_evaluations"].append({
                "task": gap,
                "evaluation": evaluation
            })

            # Count research cycle
            state["iteration"] += 1

        # Check again for missing information
        state["gaps"] = detect_gaps(
            question,
            state["evidence"]
        )

    # ==================================================
    # STEP 5: SOURCE TRACKING
    # ==================================================

    # Create source IDs for all collected research
    state["sources"] = create_source_records(
        state["search_results"]
    )


    # ==================================================
    # STEP 6: CLAIM VERIFICATION
    # ==================================================

    # Verify the collected claims
    print("\nVerifying claims...")

    state["verified_claims"] = verify_claims(
        state["evidence"]
    )


    # ==================================================
    # STEP 7: FINAL REPORT GENERATION
    # ==================================================

    # Generate final report
    print("\nGenerating final research report...")

    state["report"] = generate_report(
    state["question"],
    state["evidence"],
    state["verified_claims"],
    state["sources"]
    )
    
    return state   # Return complete agent state


# PROGRAM ENTRY POINT
# ======================================================
if __name__ == "__main__":

    # Get research question from user
    question = input("Enter your research question: ")

    # Run the autonomous research agent
    final_state = run_research_agent(question)

    print(
        "\n========== FINAL RESEARCH REPORT ==========\n"
    )

    print(final_state["report"])

    print(
        "\n========== RESEARCH SUMMARY ==========\n"
    )

    print(f"Research iterations: {final_state['iteration']}")

    print("\nRemaining gaps:")

    if final_state["gaps"]:
        for gap in final_state["gaps"]:
            print(f"- {gap}")
    else:
        print("No major gaps detected.")