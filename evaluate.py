import os
import shutil

import duckdb
import numpy as np
import pandas as pd
import pygmo as pg
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context="notebook", style="whitegrid")

from tqdm import tqdm

# This file mostly serves as an inspiration for how you can work with
# the experiment results - you probably want to modify this to analyze
# what you are interested in and possibly use jupyter notebooks instead...

RESULT_DIRS = ["results"]
PREPROCESSING_DIR = "preprocessed"
PLOT_DIR = "plots"

def preprocess(input_dirs: list[str] = RESULT_DIRS, output_dir: str = PREPROCESSING_DIR, clean: bool = False):
    """Performs preprocessing for futher analysis of the raw results, to speed up further computations."""
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        if clean:
            shutil.rmtree(output_dir)
        else:
            return
    
    print("Preprocessing...")
    with duckdb.connect(":memory:") as conn:
        conn.sql(f"CREATE OR REPLACE VIEW results AS SELECT * FROM read_csv({str([f'{d}/**/*.csv' for d in input_dirs])}, union_by_name=true)")
        conn.sql(f"COPY results TO '{output_dir}.parquet '(FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE 1)")
    print("done.")


def plot_convergence_graphs(
        y_variables=["r2_test"],
        x_variables=["generation", "evaluations", "time_seconds"],
        num_steps: int = 100,
        input_dir: str = PREPROCESSING_DIR,
        output_dir: str = PLOT_DIR,
        dpi: int = 600
):
    """Plots convergence graphs for each of the `x_variables` and `y_variables` specified with a resolution of `num_steps`."""
    with duckdb.connect(":memory:") as conn:
        conn.sql(
            f"CREATE OR REPLACE VIEW results AS SELECT * FROM read_parquet('{input_dir}*.parquet', hive_partitioning = false)")

        methods = sorted(
            [m for m, *_ in conn.sql("SELECT method FROM results GROUP BY ALL ORDER BY method ASC").fetchall()])
        problems = sorted(
            [p for p, *_ in conn.sql("SELECT problem FROM results GROUP BY ALL ORDER BY problem ASC").fetchall()])

        nrows, ncols = len(y_variables), len(x_variables) * len(problems)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(3 * ncols, 3 * nrows + 0.5),
            gridspec_kw=dict(
                wspace=0.3,
                hspace=0.3,
            ),
            squeeze=False
        )

        hues = sns.color_palette(n_colors=len(methods))
        palette = {m: hues[i] for i, m in enumerate(methods)}

        for k, y_var in enumerate(y_variables):
            progress = tqdm(desc=f"Plotting {y_var}...", total=len(x_variables) * len(problems) * num_steps)
            for i, x_var in enumerate(x_variables):
                for j, problem in enumerate(problems):
                    ax = axes[k, i * len(problems) + j]

                    df = pd.DataFrame()
                    max_x_value = \
                    conn.execute(f"SELECT MAX({x_var}::DOUBLE) FROM results WHERE problem = $1", [problem]).fetchone()[
                        0]
                    for x in np.linspace(0, max_x_value, num_steps, endpoint=True):
                        df = pd.concat([df, conn.execute(f"""
                            SELECT
                                method,
                                format('{{}}.{{}}', fold, repeat)::DOUBLE AS run,
                                {x}::DOUBLE AS {x_var},
                                {"MAX" if "r2" in y_var else "MIN"}({y_var}::DOUBLE) AS {y_var}
                            FROM results
                            WHERE problem = $1 AND {x_var}::DOUBLE <= {x}
                            GROUP BY ALL
                        """, [problem]).df()], ignore_index=True)
                        progress.update()

                    sns.lineplot(
                        df,
                        x=x_var,
                        y=y_var,
                        hue="method",
                        hue_order=methods,
                        estimator=np.median,
                        errorbar=("pi", 50),
                        err_kws=dict(lw=0),
                        legend=False,
                        ax=ax
                    )

                    ax.set_title(problem if k == 0 else "")
                    ax.set_ylabel(y_var if j == 0 else "")
                    ax.set_xlabel(x_var)

                    line_data = {}
                    for method in methods:
                        method_df = df[df['method'] == method]
                        median_data = method_df.groupby(x_var)[y_var].median().reset_index()

                        line_data[method] = {
                            'x': median_data[x_var].values,
                            'y': median_data[y_var].values
                        }

                    if 'r2' in y_var:
                        converged = []
                        for method in methods:
                            max_y = line_data[method]['y'].max()
                            min_x = line_data[method]['x'][line_data[method]['y'] >= max_y * 0.99][0]
                            converged.append(min_x)
                        threshold = np.max(converged)
                        ax.set_ylim(0, 1)
                        ax.set_xlim(0, min(threshold * 1.1, df[x_var].max()))

                    if "mse" in y_var:
                        converged = []
                        for method in methods:
                            min_y = line_data[method]['y'].min()
                            max_x = line_data[method]['x'][line_data[method]['y'] <= min_y * 1.01][0]
                            converged.append(max_x)
                        threshold = np.min(converged)
                        ax.set_xlim(0, min(threshold * 1.1, df[x_var].max()))
                        ax.set_yscale("log")

        fig.subplots_adjust(bottom=(0.3 / nrows))
        fig.legend(
            labels=methods,
            handles=[plt.plot([], [], color=palette[m])[0] for m in methods],
            ncols=len(methods),
            frameon=False,
            fancybox=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0)
        )

        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"convergence_all.png"), bbox_inches="tight", dpi=dpi)
        plt.close(fig)

#  I am sorry for this comment block we needed to quickly plot multiple metrics in the same graph
#
# def plot_convergence_graphs(
#         y_variables = ["r2_test"],
#         x_variables = ["generation", "evaluations", "time_seconds"],
#         num_steps: int = 100,
#         input_dir: str = PREPROCESSING_DIR,
#         output_dir: str = PLOT_DIR,
#         dpi: int = 600
# ):
#     """Plots convergence graphs for each of the `x_variables` and `y_variables` specified with a resolution of `num_steps`.
#
#     Note: this plot is mostly just to provide a possible starting point, for use in the report it likely is too much information in one figure.
#     """
#     with duckdb.connect(":memory:") as conn:
#         conn.sql(f"CREATE OR REPLACE VIEW results AS SELECT * FROM read_parquet('{input_dir}*.parquet', hive_partitioning = false)")
#
#         methods = sorted([m for m,*_ in conn.sql("SELECT method FROM results GROUP BY ALL ORDER BY method ASC").fetchall()])
#         problems = sorted([p for p,*_ in conn.sql("SELECT problem FROM results GROUP BY ALL ORDER BY problem ASC").fetchall()])
#
#         for y_var in y_variables:
#             progress = tqdm(desc=f"Plotting {y_var}...", total=len(x_variables) * len(problems) * num_steps)
#
#             nrows, ncols = len(x_variables), len(problems)
#             fig, axes = plt.subplots(
#                 nrows=nrows,
#                 ncols=ncols,
#                 figsize=(3 * ncols, 3 * nrows + 0.5),
#                 gridspec_kw=dict(
#                     wspace=0.3,
#                     hspace=0.3,
#                 ),
#                 squeeze=False
#             )
#
#             hues = sns.color_palette(n_colors=len(methods))
#             palette = { m:hues[i] for i,m in enumerate(methods) }
#             for i, x_var in enumerate(x_variables):
#                 for j, problem in enumerate(problems):
#                     ax = axes[i,j]
#
#                     df = pd.DataFrame()
#                     max_x_value = conn.execute(f"SELECT MAX({x_var}::DOUBLE) FROM results WHERE problem = $1", [problem]).fetchone()[0]
#                     for x in np.linspace(0, max_x_value, num_steps, endpoint=True):
#                         df = pd.concat([df, conn.execute(f"""
#                             SELECT
#                                 method,
#                                 format('{{}}.{{}}', fold, repeat)::DOUBLE AS run,
#                                 {x}::DOUBLE AS {x_var},
#                                 {"MAX" if "r2" in y_var else "MIN"}({y_var}::DOUBLE) AS {y_var}
#                             FROM results
#                             WHERE problem = $1 AND {x_var}::DOUBLE <= {x}
#                             GROUP BY ALL
#                         """, [problem]).df()], ignore_index=True)
#                         progress.update()
#
#                     sns.lineplot(
#                         df,
#                         x=x_var,
#                         y=y_var,
#                         hue="method",
#                         hue_order=methods,
#                         estimator=np.median,
#                         errorbar=("pi", 50),
#                         err_kws=dict(lw=0),
#                         # estimator=None,
#                         # units="run",
#                         legend=False,
#                         ax=ax
#                     )
#
#                     ax.set_title(problem if i == 0 else "")
#                     ax.set_ylabel(y_var if j == 0 else "")
#                     ax.set_xlabel(x_var)
#
#                     line_data = {}
#                     for method in methods:
#                         method_df = df[df['method'] == method]
#                         median_data = method_df.groupby(x_var)[y_var].median().reset_index()
#
#                         # Store the result in the dictionary
#                         line_data[method] = {
#                             'x': median_data[x_var].values,
#                             'y': median_data[y_var].values
#                         }
#
#                     # Print the line data for each method
#                     if 'r2' in y_var:
#                         converged = []
#                         for method in methods:
#                             max_y = line_data[method]['y'].max()
#                             min_x = line_data[method]['x'][line_data[method]['y'] >= max_y * 0.99][0]
#                             converged.append(min_x)
#                         threshold = np.max(converged)
#                         ax.set_xlim(0, min(threshold * 1.1, df[x_var].max()))
#
#                     if "mse" in y_var:
#                         converged = []
#                         for method in methods:
#                             min_y = line_data[method]['y'].min()
#                             max_x = line_data[method]['x'][line_data[method]['y'] <= min_y * 1.01][0]
#                             converged.append(max_x)
#                         threshold = np.min(converged)
#                         ax.set_xlim(0, max(threshold * 1.1, df[x_var].max()))
#                         ax.set_yscale("log")
#
#             fig.subplots_adjust(bottom=(0.3 / nrows))
#             if 'mse' not in y_var:
#                 fig.legend(
#                     labels=methods,
#                     handles=[plt.plot([], [], color=palette[m])[0] for m in methods],
#                     ncols=len(methods),
#                     frameon=False,
#                     fancybox=False,
#                     loc="lower center",
#                     bbox_to_anchor=(0.5, 0.0)
#                 )
#
#             os.makedirs(output_dir, exist_ok=True)
#             fig.savefig(os.path.join(output_dir, f"convergence_{y_var}.png"), bbox_inches="tight", dpi=dpi)
#             plt.close(fig)

def plot_hypervolume(
    input_dir: str = PREPROCESSING_DIR,
    output_dir: str = PLOT_DIR,
    objectives: list[str] = ("size", "mse_test"),
    ref_point: tuple[float, float] = (1.1, 1.1),
    combined_non_dominated_front: bool = False,
    dpi: int = 600
):
    """Computes the hypervolume per problem and method for the given (minimization!) objectives and reference point.
    
    Optionally combines the fronts of all method runs.
    """
    objectives = list(objectives)

    if combined_non_dominated_front:
        by = ["method"]
    else:
        by = ["method", "fold", "repeat"]

    with duckdb.connect(":memory:") as conn:
        conn.sql(f"CREATE OR REPLACE VIEW results AS SELECT * FROM read_parquet('{input_dir}*.parquet', hive_partitioning = true)")

        methods = sorted([m for m,*_ in conn.sql("SELECT method FROM results GROUP BY ALL ORDER BY method ASC").fetchall()])
        problems = sorted([p for p,*_ in conn.sql("SELECT problem FROM results GROUP BY ALL ORDER BY problem ASC").fetchall()])
        
        hues = sns.color_palette(n_colors=len(methods))
        palette = { m:hues[i] for i,m in enumerate(methods) }

        progress = tqdm(desc="Plotting hypervolume ...", total=len(problems) * len(methods))

        rows = []
        for problem in problems:
            df = conn.execute("""
                WITH
                    -- NB: only using the last rows per run may not include the best solutions
                    -- encountered in that run if the method is not elititst
                    last_rows AS (
                        SELECT
                            problem,
                            method,
                            fold,
                            repeat,
                            MAX(generation::UINTEGER) AS generation
                        FROM results
                        WHERE problem = $1
                        GROUP BY ALL
                    )
                SELECT
                    *
                FROM results INNER JOIN last_rows ON (
                    results.problem    = last_rows.problem
                    AND results.method = last_rows.method
                    AND results.fold   = last_rows.fold
                    AND results.repeat = last_rows.repeat
                    AND results.generation = last_rows.generation
                )
                GROUP BY ALL
            """, [problem]).df()

            all_objective_values = np.array(df[objectives].values.tolist())

            # normalize objectives
            obj_min = np.min(all_objective_values, axis=0, where=np.isfinite(all_objective_values), initial=np.inf)
            obj_max = np.max(all_objective_values, axis=0, where=np.isfinite(all_objective_values), initial=-np.inf)
            normed = lambda o_vals: (np.minimum(o_vals, obj_max) - obj_min) / (obj_max - obj_min + 1e-8)

            for (method,*_), method_df in df.groupby(by=by):
                objective_values = normed(np.array(method_df[objectives].values.tolist()))
                hv = pg.hypervolume(objective_values)
                
                rows.append(dict(
                    problem=problem,
                    method=method,
                    hypervolume=hv.compute(ref_point)
                ))

                progress.update()
        
        df = pd.DataFrame(rows)
        
        g = sns.catplot(
            df,
            kind="bar" if combined_non_dominated_front else "box",
            x="method",
            order=methods,
            hue="method",
            palette=palette,
            y="hypervolume",
            col="problem",
            col_order=problems
        )

        os.makedirs(output_dir, exist_ok=True)
        g.savefig(
            os.path.join(
                output_dir,
                f"hypervolume_{'combined_' if combined_non_dominated_front else ''}" \
                    + f"{'_'.join(objectives)}_{'_'.join(map(str, list(ref_point)))}.pdf"
            ),
            bbox_inches="tight",
            dpi=dpi
        )
        plt.close(g.figure)

if __name__ == "__main__":
    preprocess(clean=True)
    # What we run in the report
    plot_convergence_graphs(y_variables=["r2_test", "mse_test"], x_variables=["generation"])
    # Original call
    # plot_convergence_graphs(y_variables=["mse_train", "r2_test"])
    # plot_hypervolume()
