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
For theoretical markdown files (e.g., `Task_allocation_and_auction_mechanisms.md`, `Auction_mechanisms_for_resource_allocation.md`):

1. **Structure**:
   - Use numbered sections with clear hierarchical organization (e.g., 1, 1.1, 1.1.1)
   - Include an introduction, core content sections, and a conclusion
   - Add a comprehensive references section at the end

2. **Content Style**:
   - Educational blog/textbook tone that balances formality with accessibility
   - Begin with conceptual explanations before introducing mathematical formalism
   - Include practical examples that connect theory to robotic applications
   - Use bold text for key terms when first introduced


3. **Technical Elements**:
   - Use mathematical notation when appropriate (with LaTeX formatting)
   - Include "Mathematical Representation" subsections for formal definitions
   - Provide concrete examples with numerical values to illustrate concepts
   - Balance theoretical foundations with practical implementation considerations
   - Be careful that you are not falling into the trap of starting to write notes. Too many listings and too little information, makes it impossible to implement in code. I need it to be written clearly so i can program i afterwards.

4. **Visual Organization**:
   - Use clear subsection headers to break up content
   - Employ categorized lists (e.g., "Types of...", "Key Challenges...")
   - Create distinct "Example" paragraphs to illustrate applications
   - Highlight "Research Challenges" or "Future Directions" where relevant