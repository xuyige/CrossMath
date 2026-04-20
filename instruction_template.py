GENERAL_TASK_DESCRIPTION = """
A **Cross Math Puzzle** is a grid-based arithmetic reasoning task in which numbers and arithmetic operators are arranged in a crossword-like 2D structure. Some cells contain known numbers or symbols (`+`, `-`, `×`, `÷`, `=`), while some cells are missing and marked as `?`. Valid arithmetic equations are formed along specific **horizontal** and **vertical** paths in the grid. The objective is to infer the missing values so that **all intersecting equations are simultaneously satisfied**.

Compared with standard single-line arithmetic problems, a cross math puzzle requires the model to:
1. understand the **2D spatial structure** of the puzzle,
2. identify which cells belong to the same equation,
3. reason over **multiple dependent equations**, and
4. maintain consistency across cells shared by both horizontal and vertical equations.

"""

TEXT_ONLY_INPUT_DESCRIPTION = """
You are given a **cross math puzzle** in a **textual markdown grid format**.

Each cell in the grid may contain:
- a number,
- an arithmetic operator (`+`, `-`, `×`, `÷`),
- an equality sign (`=`),
- a missing value marked as `?`,
- or an empty cell.

The markdown table represents the **2D spatial layout** of the puzzle. A valid arithmetic equation is formed whenever numbers and operators appear continuously in a horizontal or vertical direction and include an equality sign. The missing cells marked with `?` must be filled with appropriate numbers such that **every horizontal and vertical equation is correct**.
"""

API_TEXT_ONLY_INPUT_DESCRIPTION = """
You are given a **cross math puzzle** in a **textual markdown grid format**.

Each cell in the grid may contain:
- a number,
- an arithmetic operator (`+`, `-`, `×`, `÷`),
- an equality sign (`=`),
- a missing value marked as `?`,
- or an empty cell.

The markdown table represents the **2D spatial layout** of the puzzle. A valid arithmetic equation is formed whenever numbers and operators appear continuously in a horizontal or vertical direction and include an equality sign. The missing cells marked with `?` must be filled with appropriate numbers such that **every horizontal and vertical equation is correct**.

You are required to reason as **concisely** as you can, while keeping compulsory reasoning steps.
"""

IMAGE_ONLY_INPUT_DESCRIPTION = """
You are given a **cross math puzzle** in a **image format**.

The puzzle is displayed as a 2D grid in which some cells contain numbers and arithmetic symbols (`+`, `-`, `×`, `÷`, `=`), while other cells are blank or unknown. Unknown cells correspond to values that must be inferred.

The model must first understand the **visual layout** of the puzzle: it needs to recognize the cell positions, read the symbols and numbers, determine which cells form horizontal and vertical equations, and then solve for the missing values so that **all equations in the grid are satisfied simultaneously**.
"""

HYBRID_INPUT_DESCRIPTION = """
You are given a **cross math puzzle** in **both image format and textual markdown format**.

The image provides the visual structure and appearance of the puzzle, while the markdown table provides an explicit symbolic representation of the same grid.

The model may use both modalities together:
- the **image** helps recover spatial arrangement and resolve layout ambiguity,
- the **textual grid** helps read symbols and numbers more reliably,
- combining both sources improves robustness in identifying equations and solving the puzzle.

The goal is to infer the missing values marked by `?` so that **all horizontal and vertical arithmetic equations are simultaneously satisfied**.
"""

OUTPUT_FORMAT_DESCRIPTION = """
Output Instructions:

Thinking step-by-step and output the intermediate reasoning steps.

Then provide the final answer enclosed within `<answer>` and `</answer>` tags.

- List the filled blanks in **left-to-right, top-to-bottom** order.
- Include **only** the blanks that you fill.
- Separate each number or operator with a **single space**.
- Do **not** include any additional explanation, punctuation, or formatting outside the answer tags.

"""

TASK_DESCRIPTION = """
Here are the cross math puzzle:
"""


BACKUP = """
### Output Format
`<answer>blank_1 blank_2 blank_3 ... blank_n</answer>`

### Example
If the filled blanks, read from left to right and from top to bottom, are `12`, `+`, `7`, and `35`, then the final answer should be:

`<answer>12 + 7 35</answer>`
"""
