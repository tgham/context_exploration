// Initialize jsPsych
const jsPsych = initJsPsych({
    on_finish: function() {
        jsPsych.data.displayData();
    }
});

// Define the Grid class
class Grid {
    constructor(gridData) {
        this.trialInfo = gridData.trial_info; // Array of trial info
        this.envCosts = gridData.env_costs; // Object of environment costs
        this.observedCosts = {}; // Track observed costs for each grid
        this.currentGrid = 0; // Track the current grid
    }

    // Get the binary costs for a specific grid
    getBinaryCosts(gridId) {
        return this.envCosts[gridId];
    }

    // Get the trial info for a specific trial
    getTrialInfo(trialIndex) {
        return this.trialInfo[trialIndex];
    }

    // Get the start and goal positions for a specific trial
    getStartAndGoal(trialIndex) {
        const trial = this.getTrialInfo(trialIndex);
        return {
            startA: trial.start_A,
            startB: trial.start_B,
            goalA: trial.goal_A,
            goalB: trial.goal_B
        };
    }

    // Get the paths for a specific trial
    getPaths(trialIndex) {
        const trial = this.getTrialInfo(trialIndex);
        return {
            pathA: trial.path_A,
            pathB: trial.path_B
        };
    }

    // Create the grid HTML for a specific trial
    createGridHTML(trialIndex) {
        const trial = this.getTrialInfo(trialIndex);
        const binaryCosts = this.getBinaryCosts(trial.grid);
        
        let gridHTML = `
            <div class="cost-display-container">
                <h2 class="cost-total">Total Cost:</h2>
                <p id="total-cost" class="cost-total">-$${totalCost}</p>
                <p id="trial-cost" class="cost-trial hidden">-$0</p> 
            </div>
            <div class="grid-container">
        `;
    
        for (let row = 0; row < 11; row++) {
            for (let col = 0; col < 11; col++) {
                const cellId = `cell-${row}-${col}`;
                const isStartA = row === trial.start_A[0] && col === trial.start_A[1];
                const isStartB = row === trial.start_B[0] && col === trial.start_B[1];
                const isGoalA = row === trial.goal_A[0] && col === trial.goal_A[1];
                const isGoalB = row === trial.goal_B[0] && col === trial.goal_B[1];
                const isPathA = trial.path_A.some(coord => coord[0] === row && coord[1] === col);
                const isPathB = trial.path_B.some(coord => coord[0] === row && coord[1] === col);
    
                const observedCost = this.observedCosts[`${row}-${col}`];
                const observedClass = observedCost !== undefined ? 
                    (observedCost === -1 ? 'observed-cost' : 'observed-no-cost') : '';
    
                if (isStartA) {
                    gridHTML += `<div class="grid-cell start blue-path ${observedClass}" id="${cellId}">S</div>`;
                } else if (isStartB) {
                    gridHTML += `<div class="grid-cell start green-path ${observedClass}" id="${cellId}">S</div>`;
                } else if (isGoalA) {
                    gridHTML += `<div class="grid-cell goal blue-path ${observedClass}" id="${cellId}">G</div>`;
                } else if (isGoalB) {
                    gridHTML += `<div class="grid-cell goal green-path ${observedClass}" id="${cellId}">G</div>`;
                } else if (isPathA || isPathB) {
                    const pathClass = isPathA ? 'blue-path' : 'green-path';
                    gridHTML += `<div class="grid-cell ${observedClass} ${pathClass}" id="${cellId}">★</div>`;
                } else {
                    gridHTML += `<div class="grid-cell ${observedClass}" id="${cellId}"></div>`;
                }
            }
        }
    
        gridHTML += `</div>`; // ✅ Removed choice-container from here
    
        return gridHTML;
    }
    
    
    
    
    
    
    

    // Record observed costs for a path
    recordObservedCosts(path, binaryCosts) {
        path.forEach(cell => {
            const [row, col] = cell;
            
            // Check for out-of-bounds error
            if (row < 0 || row > 10 || col < 0 || col > 10) {
                console.error(`Error in observed costs: Cell (${row}, ${col}) is out of bounds.`);
                return;
            }
    
            const cost = binaryCosts[row][col];
            this.observedCosts[`${row}-${col}`] = cost;
    
            console.log(`Recorded observed cost for cell (${row}, ${col}): ${cost}`);
        });
    }    

    // Reset the grid for a new set of trials
    resetGrid() {
        this.observedCosts = {}; 
        this.currentGrid++; 
    
        // Reset trial cost
        const trialCostElement = document.getElementById("trial-cost");
        if (trialCostElement) {
            trialCostElement.textContent = "+$0";
            trialCostElement.classList.add("hidden");
        }
    
        // Reset total cost
        totalCost = 0;
        const totalCostElement = document.getElementById("total-cost");
        if (totalCostElement) {
            totalCostElement.textContent = "-$0";
        }
    }    
    
}

// Load the JSON data and initialize the Grid class
let grid;
let currentTrialIndex = 0;

fetch('assets/expt_info.json') // Updated JSON file path
    .then(response => response.json())
    .then(data => {
        grid = new Grid(data); // Initialize the Grid class with the loaded data
        console.log('Grid data loaded:', grid);
        initializeExperiment(); // Call a function to start the experiment
    })
    .catch(error => console.error('Error loading JSON:', error));

// Function to animate the agent along the chosen path
let totalCost = 0; // Keeps track of total cost across trials
function animateAgent(path, binaryCosts, callback) {
    let currentStep = 0;
    let trialCost = 0; // Resets each trial
    let trialCostVisible = false; // Tracks visibility of red trial cost number

    function step() {
        if (currentStep > 0) {
            const [prevRow, prevCol] = path[currentStep - 1];
            const prevCellElement = document.getElementById(`cell-${prevRow}-${prevCol}`);

            if (prevCellElement) {
                prevCellElement.classList.remove('avatar');
                const cost = binaryCosts[prevRow][prevCol];

                if (cost === -1) {
                    prevCellElement.classList.add('observed-cost');
                    trialCost++; // Increment trial cost

                    // Show trial cost only when first cost is incurred
                    if (!trialCostVisible) {
                        const trialCostElement = document.getElementById("trial-cost");
                        if (trialCostElement) {
                            trialCostElement.classList.remove("hidden");
                            trialCostVisible = true;
                        }
                    }
                } else {
                    prevCellElement.classList.add('observed-no-cost');
                }

                // Update the trial cost display
                const trialCostElement = document.getElementById("trial-cost");
                if (trialCostElement) {
                    trialCostElement.textContent = `-$${trialCost}`;
                }
            }
        }

        if (currentStep < path.length) {
            const [curRow, curCol] = path[currentStep];
            const cellElement = document.getElementById(`cell-${curRow}-${curCol}`);

            if (cellElement) {
                cellElement.classList.add('avatar');
            } else {
                console.error(`Cell not found in DOM: cell-${curRow}-${curCol}`);
                return;
            }

            currentStep++;
            setTimeout(step, 300);
        } else {
            // Animation complete → Move trial cost upward into total cost
            mergeCosts(trialCost, callback);
        }
    }

    setTimeout(step, 300);
}




//  merge costs together 
function mergeCosts(trialCost, callback) {
    const totalCostElement = document.getElementById("total-cost");
    const trialCostElement = document.getElementById("trial-cost");

    if (totalCostElement && trialCostElement) {
        trialCostElement.style.transition = "transform 0.5s ease-in-out";
        trialCostElement.style.transform = "translateY(-20px)"; // Moves up

        setTimeout(() => {
            totalCost += trialCost; // Add to total cost
            totalCostElement.textContent = `-$${totalCost}`;

            // Reset trial cost display
            trialCostElement.textContent = `+$0`;
            trialCostElement.style.transform = "translateY(0)"; // Reset position
            trialCostElement.classList.add("hidden"); // Hide again
        }, 500);
    }

    setTimeout(() => {
        currentTrialIndex++; // Move to next trial
        jsPsych.finishTrial();
    }, 2000);
}









// Instructions
const instructions = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <h2>Path Selection Task</h2>
        <p>In this experiment, you will see a grid with two colored paths.</p>
        <p>Each path has a start point (S) and a goal point (G).</p>
        <p>Your task is to choose one of these paths using the arrow keys:</p>
        <p>- Press the <strong>left arrow</strong> to choose the blue path</p>
        <p>- Press the <strong>right arrow</strong> to choose the green path</p>
        <p>After you choose, you'll see an avatar move along the path.</p>
        <p>Some states will have a cost ($), and others will be free (✓).</p>
        <p>Press any key to begin.</p>
    `,
    choices: "ALL_KEYS"
};

// New grid message
const newGridMessage = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return `
            <h2>New Grid</h2>
            <p>You are now in a new grid. Press any key to start.</p>
        `;
    },
    choices: "ALL_KEYS",
    on_finish: function() {
        grid.resetGrid(); // Reset the grid for the new set of trials
    }
};

// Path selection trial
const pathSelectionTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return `
            ${grid.createGridHTML(currentTrialIndex)}
            <div class="choice-container">
                <div class="choice-box blue-path" id="blue-choice">Left Arrow: <strong>Blue Path</strong></div>
                <div class="choice-box green-path" id="green-choice">Right Arrow: <strong>Green Path</strong></div>
            </div>
        `;
    },
    choices: ['ArrowLeft', 'ArrowRight'], 
    on_finish: function(data) {
        if (data.response === 'ArrowLeft') {
            data.choice = 'blue';
        } else if (data.response === 'ArrowRight') {
            data.choice = 'green';
        } else {
            console.warn(`Invalid key pressed: ${data.response}. Waiting for valid input.`);
            return false; 
        }
        jsPsych.data.get().addToLast({ choice: data.choice });
    }
};

// Path animation trial
const pathAnimationTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const lastTrialData = jsPsych.data.get().last(1).values()[0];
        const chosenPath = lastTrialData.choice === 'blue' ? 'blue-choice' : 'green-choice';
        const unchosenPath = lastTrialData.choice === 'blue' ? 'green-choice' : 'blue-choice';
        
        return `
            ${grid.createGridHTML(currentTrialIndex)}
            <div class="choice-container">
                <div class="choice-box blue-path" id="blue-choice" style="${lastTrialData.choice === 'blue' ? '' : 'visibility: hidden;'}">Left Arrow: <strong>Blue Path</strong></div>
                <div class="choice-box green-path" id="green-choice" style="${lastTrialData.choice === 'green' ? '' : 'visibility: hidden;'}">Right Arrow: <strong>Green Path</strong></div>
            </div>
        `;
    },
    choices: "NO_KEYS",
    on_load: function() {
        const currentTrial = grid.getTrialInfo(currentTrialIndex);
        const lastTrialData = jsPsych.data.get().last(1).values()[0];

        if (!lastTrialData || !lastTrialData.choice) {
            console.error("No valid path choice found. Restarting trial.");
            return jsPsych.finishTrial();
        }

        const chosenPath = lastTrialData.choice === 'blue' ? currentTrial.path_A : currentTrial.path_B;
        const binaryCosts = grid.getBinaryCosts(currentTrial.grid);

        console.log("Animating Trial:", currentTrialIndex);
        console.log("Chosen Path:", lastTrialData.choice, chosenPath);

        grid.recordObservedCosts(chosenPath, binaryCosts);

        setTimeout(() => {
            animateAgent(chosenPath, binaryCosts, function() {
                jsPsych.finishTrial();
            });
        }, 100);
    }
};

// Results summary
const resultsSummary = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const lastTrialData = jsPsych.data.get().last(2).values()[0];
        const currentTrial = grid.getTrialInfo(currentTrialIndex);
        const chosenPath = lastTrialData.choice === 'blue' ? currentTrial.path_A : currentTrial.path_B;
        const binaryCosts = grid.getBinaryCosts(currentTrial.grid);

        let totalCost = 0;
        chosenPath.forEach(([r, c]) => {
            if (binaryCosts[r][c] === -1) totalCost++;
        });

        // <h2>Results Summary</h2>
        // <p>You chose the <strong>${lastTrialData.choice}</strong> path.</p>
        // <p>Total cost: <strong>${totalCost}</strong> states</p>
        return `
            <p>Press any key to continue.</p>
        `;
    },
    choices: "ALL_KEYS", // **Requires explicit keypress**
    trial_duration: null, // **Ensures no automatic skipping**
    on_finish: function() {
        currentTrialIndex++; // Move to the next trial only after keypress
    }
};



// End message
const end = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <p>Thank you for participating!</p>
        <p>The experiment is now complete.</p>
        <p>Press any key to see your data.</p>
    `,
    choices: "ALL_KEYS"
};

// Create timeline
function createTimeline() {
    const timeline = [instructions];

    // Loop through all trials and add them to the timeline
    for (let i = 0; i < grid.trialInfo.length; i++) {
        if (i % 4 === 0 && i !== 0) {
            // Add new grid message every 4 trials
            timeline.push(newGridMessage);
        }
        timeline.push(pathSelectionTrial);
        timeline.push(pathAnimationTrial); 
    }

    // Add the end message
    timeline.push(end);

    return timeline;
}


// Start experiment when the page loads
function initializeExperiment() {
    const timeline = createTimeline();
    jsPsych.run(timeline);
}