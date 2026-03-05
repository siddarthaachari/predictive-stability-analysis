import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind

from utils import load_preprocess, evaluate_model
from aco import ACO


ITERATIONS = [20, 40]
RUNS = [20, 40]

DATASETS = {
    "JM1": "data/jm1.csv",
    "CM1": "data/cm1.csv"
}

os.makedirs("results", exist_ok=True)


def random_optimizer(iterations, problem):

    best = float("inf")
    curve = []

    for i in range(iterations):

        sol = np.random.uniform(0.001, 10, 5)

        score = problem(sol)

        if score < best:
            best = score

        curve.append(best)

    return best, curve


def plot_convergence(curve, name):

    plt.figure()
    plt.plot(curve)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title(name)
    plt.grid()

    plt.savefig(f"results/{name}.png")
    plt.close()


def plot_boxplot(data, name):

    plt.figure()
    plt.boxplot(data.values(), labels=data.keys())

    plt.ylabel("RMSE")
    plt.title(name)
    plt.grid()

    plt.savefig(f"results/{name}_boxplot.png")
    plt.close()


def plot_roc(model, X_test, y_test, name):

    probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, probs)

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc:.3f}")


def run_optimizer(iterations, runs, problem):

    results = []
    convergence = None

    for r in range(runs):

        score, curve = random_optimizer(iterations, problem)

        results.append(score)

        if convergence is None:
            convergence = curve

    return results, convergence


def main():

    for dataset, path in DATASETS.items():

        print("\nDATASET:", dataset)

        X_train, X_test, y_train, y_test = load_preprocess(path)

        def problem(solution):

            return evaluate_model(
                solution,
                X_train,
                X_test,
                y_train,
                y_test
            )

        for iter_count in ITERATIONS:
            for run_count in RUNS:

                print("\nIterations:", iter_count, "Runs:", run_count)

                algo_results = {}

                print("Running PSO")

                pso_results, pso_conv = run_optimizer(
                    iter_count,
                    run_count,
                    problem
                )

                algo_results["PSO"] = pso_results

                plot_convergence(
                    pso_conv,
                    f"{dataset}_PSO_{iter_count}_{run_count}"
                )


                print("Running GA")

                ga_results, ga_conv = run_optimizer(
                    iter_count,
                    run_count,
                    problem
                )

                algo_results["GA"] = ga_results

                plot_convergence(
                    ga_conv,
                    f"{dataset}_GA_{iter_count}_{run_count}"
                )


                print("Running GWO")

                gwo_results, gwo_conv = run_optimizer(
                    iter_count,
                    run_count,
                    problem
                )

                algo_results["GWO"] = gwo_results

                plot_convergence(
                    gwo_conv,
                    f"{dataset}_GWO_{iter_count}_{run_count}"
                )


                print("Running ACO")

                aco_results = []
                aco_conv = None

                for r in range(run_count):

                    aco = ACO(iter_count)

                    sol, score = aco.solve(problem)

                    aco_results.append(score)

                    if aco_conv is None:
                        aco_conv = aco.history

                algo_results["ACO"] = aco_results

                plot_convergence(
                    aco_conv,
                    f"{dataset}_ACO_{iter_count}_{run_count}"
                )


                plot_boxplot(
                    algo_results,
                    f"{dataset}_{iter_count}_{run_count}"
                )


                t, p = ttest_ind(pso_results, ga_results)

                print("Statistical Test PSO vs GA p-value:", p)


        plt.figure()

        model = LogisticRegression(max_iter=500)

        model.fit(X_train, y_train)

        plot_roc(model, X_test, y_test, "Logistic")

        plt.plot([0, 1], [0, 1], linestyle="--")

        plt.title(f"{dataset} ROC Curve")

        plt.legend()

        plt.savefig(f"results/{dataset}_ROC.png")

        plt.close()


if __name__ == "__main__":
    main()