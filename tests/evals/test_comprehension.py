from contrastive_sdf.evals.scoring.iteration_style import classify_iteration
from tests.evals.test_iteration_feature import POLICY


def observations():
    data = []
    for b in ("A", "B"):
        for readout in ("semantic", "open_ended"):
            for authority in ("grader", "users"):
                data.append(
                    {
                        "branch": b,
                        "readout": readout,
                        "task_id": f"{readout}_{authority}",
                        "repetition": 1,
                        "authority": authority,
                        "belief": {"correct": True, "valid": True},
                    }
                )
        for world in ("A", "B"):
            for authority in ("grader", "users"):
                data.append(
                    {
                        "branch": b,
                        "readout": "comprehension_vs_loop",
                        "task_id": f"{world}_{authority}_one",
                        "repetition": 1,
                        "world": world,
                        "authority": authority,
                        "qualification": {"correct": True, "valid": True},
                    }
                )
        for task in ("one", "two"):
            for repetition in (1, 2, 3):
                source = (
                    "x=[v for v in values]"
                    if (b == "A") == (task == "one")
                    else "for v in values:\n    result.append(v)"
                )
                data.append(
                    {
                        "branch": b,
                        "readout": "behavior",
                        "task_id": task,
                        "repetition": repetition,
                        "classification": classify_iteration(source, POLICY).describe(),
                    }
                )
    return data
