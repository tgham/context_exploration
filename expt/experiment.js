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
        console.log('Trial number:', trialIndex); // Debugging: Log the trial number
        const binaryCosts = this.getBinaryCosts(trial.grid);
        let gridHTML = '<div class="grid-container">';

        // Create empty 11x11 grid
        for (let row = 0; row < 11; row++) {
            for (let col = 0; col < 11; col++) {
                const cellId = `cell-${row}-${col}`;
                const isStartA = row === trial.start_A[0] && col === trial.start_A[1];
                const isStartB = row === trial.start_B[0] && col === trial.start_B[1];
                const isGoalA = row === trial.goal_A[0] && col === trial.goal_A[1];
                const isGoalB = row === trial.goal_B[0] && col === trial.goal_B[1];
                const isPathA = trial.path_A.some(coord => coord[0] === row && coord[1] === col);
                const isPathB = trial.path_B.some(coord => coord[0] === row && coord[1] === col);

                // Check if this cell has been observed
                const observedCost = this.observedCosts[`${row}-${col}`];
                const observedClass = observedCost !== undefined ? (observedCost === -1 ? 'observed-cost' : 'observed-no-cost') : '';

                if (isStartA || isStartB) {
                    // Start state
                    gridHTML += `<div class="grid-cell start ${observedClass}" id="${cellId}">S</div>`;
                } else if (isGoalA || isGoalB) {
                    // Goal state
                    gridHTML += `<div class="grid-cell goal ${observedClass}" id="${cellId}">G</div>`;
                } else if (isPathA || isPathB) {
                    // Path cell
                    const pathClass = isPathA ? 'blue-path' : 'green-path';
                    gridHTML += `<div class="grid-cell ${observedClass} ${pathClass}" id="${cellId}">★</div>`;
                } else if (observedCost !== undefined) {
                    // Observed cell
                    if (observedCost === -1) {
                        gridHTML += `<div class="grid-cell observed-cost" id="${cellId}"></div>`;
                    } else {
                        gridHTML += `<div class="grid-cell observed-no-cost" id="${cellId}"></div>`;
                    }
                } else {
                    // Default cell
                    gridHTML += `<div class="grid-cell" id="${cellId}"></div>`;
                }
            }
        }

        gridHTML += '</div>';
        return gridHTML;
    }

    // Record observed costs for a path
    recordObservedCosts(path, binaryCosts) {
        path.forEach(cell => {
            const [row, col] = cell;
            const cost = binaryCosts[row][col];
            this.observedCosts[`${row}-${col}`] = cost;
        });
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
function animateAgent(path, binaryCosts, callback) {
    let currentStep = 0;
    const totalSteps = path.length;

    // Animation function
    function step() {
        if (currentStep > 0) {
            // Mark previous cell as visited with cost indicator
            const prevCell = path[currentStep - 1];
            const prevCellElement = document.getElementById(`cell-${prevCell[0]}-${prevCell[1]}`);
            if (prevCellElement) {
                prevCellElement.classList.remove('avatar');

                const cost = binaryCosts[prevCell[0]][prevCell[1]];
                if (cost === -1) {
                    prevCellElement.classList.add('observed-cost'); // Mark as red if cost
                } else {
                    prevCellElement.classList.add('observed-no-cost'); // Mark as grey if no cost
                }
            } else {
                console.error(`Cell not found: cell-${prevCell[0]}-${prevCell[1]}`);
            }
        }

        if (currentStep < totalSteps) {
            // Move agent to current cell
            const currentCell = path[currentStep];
            const cellElement = document.getElementById(`cell-${currentCell[0]}-${currentCell[1]}`);
            if (cellElement) {
                cellElement.classList.add('avatar');
            } else {
                console.error(`Cell not found: cell-${currentCell[0]}-${currentCell[1]}`);
            }

            currentStep++;
            setTimeout(step, 500); // 0.5 second per step
        } else {
            // Animation complete
            if (callback) callback();
        }
    }

    // Start animation
    setTimeout(step, 500); // Start after a short delay
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

// Path selection trial
const pathSelectionTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const gridHTML = grid.createGridHTML(currentTrialIndex);
        return `
            ${gridHTML}
            <div class="choice-container">
                <div class="choice-box blue-path">Left Arrow: Blue Path</div>
                <div class="choice-box green-path">Right Arrow: Green Path</div>
            </div>
        `;
    },
    choices: ['ArrowLeft', 'ArrowRight'],
    prompt: "",
    data: {
        task: 'path_selection',
        trialIndex: function() {
            return currentTrialIndex;
        }
    },
    on_finish: function(data) {
        if (data.response === 'ArrowLeft') {
            data.choice = 'blue';
        } else if (data.response === 'ArrowRight') {
            data.choice = 'green';
        }
    }
};

// Path animation trial
const pathAnimationTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return grid.createGridHTML(currentTrialIndex);
    },
    choices: "NO_KEYS",
    trial_duration: null, // Will be set dynamically
    on_load: function() {
        const currentTrial = grid.getTrialInfo(currentTrialIndex);
        const chosenPath = jsPsych.data.get().last(1).values()[0].choice === 'blue' ? currentTrial.path_A : currentTrial.path_B;
        const binaryCosts = grid.getBinaryCosts(currentTrial.grid);

        // Record observed costs
        grid.recordObservedCosts(chosenPath, binaryCosts);

        // Calculate animation duration (0.5s per step + buffer)
        const animationDuration = (chosenPath.length * 500) + 1000;

        // Set trial duration correctly
        jsPsych.getCurrentTrial().trial_duration = animationDuration;

        // Start animation
        animateAgent(chosenPath, binaryCosts, function() {
            // Animation complete, end trial after a short delay
            setTimeout(function() {
                jsPsych.finishTrial();
            }, 1000);
        });
    },
    data: function() {
        return {
            trialIndex: currentTrialIndex
        };
    }
};

// Results summary
const resultsSummary = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const last_trial = jsPsych.data.get().last(2).values()[0]; // Get the path selection trial
        const currentTrial = grid.getTrialInfo(currentTrialIndex);
        const chosenPath = last_trial.choice === 'blue' ? currentTrial.path_A : currentTrial.path_B;

        // Calculate total cost
        const totalCost = chosenPath.filter(cell => cell.cost).length;

        return `
            <h2>Results Summary</h2>
            <p>You chose the ${last_trial.choice} path.</p>
            <p>Total cost: ${totalCost} states</p>
            <p>Press any key to continue.</p>
        `;
    },
    choices: "ALL_KEYS",
    on_finish: function() {
        currentTrialIndex++; // Increment the trial index after the results summary
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
        timeline.push(pathSelectionTrial);
        timeline.push(pathAnimationTrial);
        timeline.push(resultsSummary);
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