# Game Theory Course Environment

## Commands
- **Run Scripts**: `python lesson1/intersection_game.py`
- **Install Dependencies**: `pip install -r requirements.txt`
- **Activate Environment**: `source activate.sh` or `. activate.fish` (fish shell)

## Running Games by Lesson
- Lesson 1: `python lesson1/intersection_game.py`
- Lesson 2: `python lesson2/lane_changing_game.py`
- Lesson 3: `python lesson3/roundabout_game.py`
- Lesson 4: `python lesson4/fleet_coordination.py`

## Code Style Guidelines
- **Imports**: Standard library first, third-party next, local modules last
- **Type Hints**: Use Python type annotations for function parameters and returns
- **Naming**: snake_case for variables/functions, CamelCase for classes
- **Docstrings**: Use triple quotes with Args/Returns sections
- **Error Handling**: Use try/except with specific exceptions
- **Line Length**: Keep lines under 100 characters
- **Visualization**: Prefer matplotlib for analysis plots, pygame for simulations

## Theoretical Material Writing Style
For theoretical markdown files:

1. **Document Structure**:
   - Begin with an engaging introduction that establishes the topic's importance and outlines the chapter
   - Use numbered sections with clear hierarchical organization (e.g., 1, 1.1, 1.1.1)
   - Include 4-7 major sections with 3-5 subsections each for comprehensive coverage
   - Add an insightful conclusion that summarizes key insights and future directions
   - Include a comprehensive references section with 10-15 relevant academic sources

2. **Content Depth and Progression**:
   - Create textbook-quality content with 20-30 pages of comprehensive material
   - Layer information: start with intuitive explanations, then add technical details, followed by practical applications
   - Balance theoretical foundations (60%) with implementation considerations (40%)
   - Ensure each concept builds logically on previous material
   - Include "Why This Matters" context for each major concept

3. **Mathematical Content**:
   - Include dedicated "Mathematical Representation" subsections for each key concept
   - Present complete mathematical formulations with clearly defined variables and parameters
   - Follow each formula with plain-language explanation of its components and significance
   - Provide step-by-step worked examples demonstrating practical application of formulas
   - Include parameter values, constraints, and boundary conditions for implementation

4. **Examples and Applications**:
   - Create detailed examples that progress from problem statement to complete solution
   - Include numerical values and concrete scenarios to ground abstract concepts
   - Develop multi-stage examples that show how concepts connect to real implementation
   - Add "Implementation Considerations" sections highlighting practical challenges
   - Use autonomous vehicle and multi-robot scenarios that directly connect to coding exercises

5. **Visual Organization and Readability**:
   - Structure content with consistent and clear hierarchical headings
   - Use bullet points strategically for lists, but develop full paragraphs for explanations
   - Include "Key Insights" summary boxes after complex sections
   - Employ "Example" and "Case Study" labeled sections with indentation or formatting
   - Use bold text for important terms and concepts when first introduced

6. **Programming-Ready Content**:
   - Include pseudocode for key algorithms and procedures
   - Specify data structures and their relationships explicitly
   - Define clear input-output relationships for functions and components
   - Provide parameter ranges, default values, and edge cases
   - Include complexity analysis and performance considerations

7. **Narrative Engagement**:
   - Use an educational textbook tone that balances academic rigor with accessibility
   - Address the reader directly with occasional "you" statements for engagement
   - Incorporate thought experiments and intuitive analogies for complex concepts
   - Present contrasting approaches with trade-offs to promote critical thinking
   - Include historical context and development of key ideas where relevant