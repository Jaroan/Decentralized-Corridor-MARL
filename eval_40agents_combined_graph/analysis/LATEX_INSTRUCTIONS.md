# How to Add Route Speed Analysis to Your LaTeX Paper

## Files Generated

1. **route_speeds_table.tex** - LaTeX table code
2. **route_speeds_table_figures.tex** - LaTeX figure code
3. **route_speeds_boxplot.pdf** - Box plot figure
4. **speed_vs_distance.pdf** - Scatter plot figure

## Step 1: Add Table to Paper

Copy the contents of `route_speeds_table.tex` into your paper where you want the table to appear, or use `\input{}`:

```latex
\section{Results}

\subsection{Route Performance Analysis}

Table~\ref{table:route_speeds} presents the average speed statistics for each of the 8 routes in the simple\_combined\_graph scenario with 40 agents.

% Option 1: Input the file directly
\input{eval_40agents_combined_graph/analysis/route_speeds_table.tex}

% Option 2: Or copy-paste the table code directly into your .tex file
```

## Step 2: Add Figures to Paper

Copy the contents of `route_speeds_table_figures.tex` into your paper:

```latex
\subsection{Speed Distribution by Route}

Figure~\ref{fig:route_speeds_boxplot} shows the distribution of agent speeds across all 8 routes. Route 5 (CL$\rightarrow$T17) exhibits notably high variability with a standard deviation of 29.9 knots, and contains a significant outlier at 87.3 knots (50\% of maximum speed), indicating a severe bottleneck or conflict zone.

% Option 1: Input the file directly
\input{eval_40agents_combined_graph/analysis/route_speeds_table_figures.tex}

% Option 2: Or copy-paste the figure code directly
```

## Step 3: Reference in Text

When discussing results, reference the table and figures:

```latex
As shown in Table~\ref{table:route_speeds}, route performance varies significantly
across the 8 available paths. The most efficient route, Route 3 (B$\rightarrow$T17),
achieves an average speed of 156.2 knots with low variability (std=3.7 knots),
representing 89.3\% of the theoretical maximum speed.

In contrast, Route 5 (CL$\rightarrow$T17) exhibits the poorest performance with an
average speed of only 138.4 knots and high variability (std=29.9 knots).
Figure~\ref{fig:route_speeds_boxplot} illustrates this disparity, revealing a
significant outlier at 87.3 knots, suggesting that agents on this route encounter
severe conflicts or spacing violations.

Figure~\ref{fig:speed_vs_distance} demonstrates the relationship between total
distance traveled and achieved speed. Notably, longer routes do not necessarily
result in lower speeds; Routes 6 and 7 (CR entry) achieve competitive speeds
despite traveling over 21 km.
```

## Step 4: Ensure PDF Files Are Accessible

Make sure the PDF files are in the correct directory relative to your main .tex file:

```bash
# If your main .tex file is in the root directory:
# Make sure these paths are correct in the \includegraphics commands:
#   eval_40agents_combined_graph/analysis/route_speeds_boxplot.pdf
#   eval_40agents_combined_graph/analysis/speed_vs_distance.pdf

# Or copy the PDFs to your paper's figures directory:
cp eval_40agents_combined_graph/analysis/route_speeds_boxplot.pdf paper/figures/
cp eval_40agents_combined_graph/analysis/speed_vs_distance.pdf paper/figures/

# Then update the paths in the figure code:
\includegraphics[width=0.8\textwidth]{figures/route_speeds_boxplot.pdf}
\includegraphics[width=0.8\textwidth]{figures/speed_vs_distance.pdf}
```

## Step 5: Compile Your Paper

```bash
pdflatex your_paper.tex
bibtex your_paper
pdflatex your_paper.tex
pdflatex your_paper.tex
```

## Example Complete Section

Here's a complete example of how a results section might look:

```latex
\section{Results}

\subsection{Route Performance Analysis}

We evaluated the performance of 40 agents navigating through the
simple\_combined\_graph scenario over 5 episodes. The corridor network
provides 8 distinct route options, varying by entry point (A, B, CL, CR)
and exit corridor (T16 left, T17 right).

Table~\ref{table:route_speeds} summarizes the speed statistics for each route.
Overall, agents achieved a mean speed of 150.3 knots, representing 85.9\%
of the theoretical maximum speed of 175 knots. However, performance varies
significantly across routes, with average speeds ranging from 138.4 to 156.2 knots.

\input{eval_40agents_combined_graph/analysis/route_speeds_table.tex}

The most efficient route, Route 3 (B$\rightarrow$T17), achieves an average
speed of 156.2 knots with remarkably low variability (std=3.7 knots). This
route also benefits from a relatively short corridor distance of 8.6 km and
moderate approach distance, resulting in an average completion time of 195.8 seconds.

Conversely, Route 5 (CL$\rightarrow$T17) exhibits the poorest performance,
with an average speed of 138.4 knots and exceptionally high variability
(std=29.9 knots). Figure~\ref{fig:route_speeds_boxplot} clearly shows this
route contains a severe outlier at 87.3 knots, representing a 50\% reduction
from maximum speed. This suggests agents on Route 5 encounter significant
conflicts or bottlenecks, likely at the merge point before reaching the
right exit corridor (T17).

\input{eval_40agents_combined_graph/analysis/route_speeds_table_figures.tex}

Figure~\ref{fig:speed_vs_distance} illustrates the relationship between
total distance traveled and achieved speed. Interestingly, route length
alone does not determine performance. Routes 6 and 7 (CR entry), despite
requiring over 21 km of travel, achieve competitive speeds of 153-154 knots.
This suggests that the corridor network geometry and conflict patterns,
rather than distance alone, are the primary determinants of throughput.

These results highlight the importance of route planning and conflict
resolution in multi-agent corridor navigation. The identification of
Route 5's bottleneck presents an opportunity for targeted intervention
or corridor redesign to improve overall system throughput.
```

## Tips

1. **Adjust figure width**: Change `0.8\textwidth` to fit your paper's layout (e.g., `0.6\textwidth` for narrower figures)

2. **Side-by-side figures**: To place both plots side by side:
```latex
\begin{figure}[htbp]
\centering
\begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{route_speeds_boxplot.pdf}
    \caption{Speed distribution by route}
    \label{fig:boxplot}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{speed_vs_distance.pdf}
    \caption{Speed vs distance}
    \label{fig:scatter}
\end{subfigure}
\caption{Route speed analysis for 40 agents.}
\label{fig:route_speeds}
\end{figure}
```

3. **Table positioning**: Add `[H]` to force table placement:
```latex
\usepackage{float}  % In preamble
\begin{table}[H]    % In document
```

4. **Font sizes**: Adjust table font if needed:
```latex
\begin{table}[htbp]
\centering
\small  % or \footnotesize for smaller text
\caption{...}
...
\end{table}
```
