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
                <h2 class="cost-total">Ship Damage:</h2>
                <p id="total-cost" class="cost-total">${totalCost} Units</p>
                <p id="trial-cost" class="cost-trial hidden">+0 Units</p> 
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
                    gridHTML += `<div class="grid-cell ${observedClass} ${pathClass}" id="${cellId}">⚝</div>`;
                } else {
                    gridHTML += `<div class="grid-cell ${observedClass}" id="${cellId}"></div>`;
                }
            }
        }
    
        gridHTML += `</div>`;
    
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
            trialCostElement.textContent = "+0 Units";
            trialCostElement.classList.add("hidden");
        }
    
        // Reset total cost
        totalCost = 0;
        const totalCostElement = document.getElementById("total-cost");
        if (totalCostElement) {
            totalCostElement.textContent = "0 Units";
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

// 1. Add spaceship character with animation
function createAvatar() {
    return `
        <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2L4 10l8 2 8-2z"/>
            <path d="M4 10l1 10h14l1-10"/>
            <path d="M12 12v8"/>
            <circle cx="8" cy="16" r="1"/>
            <circle cx="16" cy="16" r="1"/>
        </svg>
    `;
}

function updateCellAppearance(cell, cost) {
    if (cost === "high") {
        cell.classList.add("observed-cost");
        cell.classList.remove("observed-no-cost");
    } else if (cost === "low") {
        cell.classList.add("observed-no-cost");
        cell.classList.remove("observed-cost");
    }

    // Ensure start and goal cells are updated correctly
    if (cell.classList.contains("start")) {
        cell.style.backgroundColor = cost === "high" ? "#c53030" : "#48bb78"; // Red for high, green for low
    } 
    if (cell.classList.contains("goal")) {
        cell.style.backgroundColor = cost === "high" ? "#c53030" : "#48bb78";
    }
}


// 2. Add visual and audio feedback for costs
function animateAgent(path, binaryCosts, callback) {
    let currentStep = 0;
    let trialCost = 0;
    let trialCostVisible = false;

    function step() {
        if (currentStep > 0) {
            const [prevRow, prevCol] = path[currentStep - 1];
            const prevCellElement = document.getElementById(`cell-${prevRow}-${prevCol}`);

            if (prevCellElement) {
                prevCellElement.classList.remove('avatar');
                const cost = binaryCosts[prevRow][prevCol];

                if (cost === -1) {
                    prevCellElement.classList.add('observed-cost');
                    trialCost++;
                    
                    // Visual burst effect for radiation damage
                    prevCellElement.innerHTML += '<div class="cost-burst">+1 Damage</div>';
                    setTimeout(() => {
                        const burst = prevCellElement.querySelector('.cost-burst');
                        if (burst) burst.remove();
                    }, 600);
                    
                    // Play radiation sound (create new instance for each step)
                    const radiationSound = new Audio('assets/radiationSound.mp3');
                    radiationSound.play();

                    if (!trialCostVisible) {
                        const trialCostElement = document.getElementById("trial-cost");
                        if (trialCostElement) {
                            trialCostElement.classList.remove("hidden");
                            trialCostVisible = true;
                        }
                    }
                } else {
                    prevCellElement.classList.add('observed-no-cost');
                    
                    // Visual feedback for safe passage
                    prevCellElement.innerHTML += '<div class="free-burst">Safe</div>';
                    setTimeout(() => {
                        const burst = prevCellElement.querySelector('.free-burst');
                        if (burst) burst.remove();
                    }, 600);
                    
                    // Play safe passage sound (create new instance for each step)
                    const safeSound = new Audio('assets/safeSound.mp3');
                    safeSound.play();
                }

                const trialCostElement = document.getElementById("trial-cost");
                if (trialCostElement) {
                    trialCostElement.textContent = `+${trialCost} Units`;
                }
            }
        }

        if (currentStep < path.length) {
            const [curRow, curCol] = path[currentStep];
            const cellElement = document.getElementById(`cell-${curRow}-${curCol}`);

            if (cellElement) {
                cellElement.classList.add('avatar');
                cellElement.innerHTML = createAvatar(); // Add spaceship avatar
            } else {
                console.error(`Cell not found in DOM: cell-${curRow}-${curCol}`);
                return;
            }

            currentStep++;
            setTimeout(step, 500);
        } else {
            // Animation complete
            mergeCosts(trialCost, callback);
        }
    }

    setTimeout(step, 500);
}

// 4. Add animated transitions between trials
function mergeCosts(trialCost, callback) {
    const totalCostElement = document.getElementById("total-cost");
    const trialCostElement = document.getElementById("trial-cost");

    if (totalCostElement && trialCostElement) {
        // Add warning animation to cost display
        if (trialCost > 0) {
            trialCostElement.classList.add("cost-animate");
        }
        
        trialCostElement.style.transition = "transform 0.5s ease-in-out";
        trialCostElement.style.transform = "translateY(-20px)";

        setTimeout(() => {
            totalCost += trialCost;
            
            // Animated counter for total cost
            const startCost = totalCost - trialCost;
            const duration = 1000;
            const frameDuration = 1000/60;
            const totalFrames = Math.round(duration/frameDuration);
            let frame = 0;
            
            const counter = setInterval(() => {
                frame++;
                const progress = frame/totalFrames;
                const currentCount = Math.floor(startCost + progress * trialCost);
                totalCostElement.textContent = `${currentCount} Units`;
                
                if (frame === totalFrames) {
                    clearInterval(counter);
                    totalCostElement.textContent = `${totalCost} Units`;
                    
                    // Reset trial cost display with animation
                    trialCostElement.textContent = `+0 Units`;
                    trialCostElement.classList.remove("cost-animate");
                    trialCostElement.style.transform = "translateY(0)";
                    trialCostElement.classList.add("hidden");
                }
            }, frameDuration);
        }, 500);
    }

    setTimeout(() => {
        // Add transition effect between trials
        document.querySelector(".grid-container").classList.add("fade-transition");
        
        setTimeout(() => {
            currentTrialIndex++;
            jsPsych.finishTrial();
            
            setTimeout(() => {
                const grid = document.querySelector(".grid-container");
                if (grid) grid.classList.remove("fade-transition");
            }, 100);
        }, 500);
    }, 1500);
}

// 5. Update the path selection trial to include space theme elements
const pathSelectionTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return `
            <div class="theme-context">Choose your flight path through the asteroid field!</div>
            ${grid.createGridHTML(currentTrialIndex)}
            <div class="choice-container">
                <div class="choice-box blue-path" id="blue-choice">
                    <div class="choice-icon">🚀</div>
                    <div>Left Arrow: <strong>Blue Route</strong></div>
                </div>
                <div class="choice-box green-path" id="green-choice">
                    <div class="choice-icon">🛸</div>
                    <div>Right Arrow: <strong>Green Route</strong></div>
                </div>
            </div>
        `;
    },
    choices: ['ArrowLeft', 'ArrowRight'], 
    on_finish: function(data) {
        // Add "swipe" effect on selection
        const choice = data.response === 'ArrowLeft' ? 'blue' : 'green';
        const choiceElement = document.getElementById(choice === 'blue' ? 'blue-choice' : 'green-choice');
        const unchosenElement = document.getElementById(choice === 'blue' ? 'green-choice' : 'blue-choice');
        
        if (choiceElement && unchosenElement) {
            choiceElement.classList.add('choice-selected');
            unchosenElement.classList.add('choice-unselected');
        }
        
        data.choice = choice;
        jsPsych.data.get().addToLast({ choice: data.choice });
    }
};

// Instructions
const instructions = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <h1>Space Explorer Mission</h1>
        <p>Welcome, Space Explorer! Your mission is to navigate through dangerous asteroid fields.</p>
        
        <div class="instruction-section">
            <h2>Mission Briefing:</h2>
            <p>For each mission, you'll see two possible flight paths marked with stars:</p>
            <p>- <span class="blue-text">Blue stars</span> mark the first route</p>
            <p>- <span class="green-text">Green stars</span> mark the second route</p>
            <p>Each path has a starting point (S) and a destination (G).</p>
        </div>
        
        <div class="instruction-section">
            <h2>Your Task:</h2>
            <p>Choose which route to fly using your arrow keys:</p>
            <p>- Press <strong>LEFT ARROW</strong> to follow the blue route</p>
            <p>- Press <strong>RIGHT ARROW</strong> to follow the green route</p>
        </div>
        
        <div class="instruction-section">
            <h2>Radiation Zones:</h2>
            <p>Some sectors contain dangerous radiation that will damage your ship:</p>
            <p>- <span class="red-text">Red sectors</span> are radiation zones that cause 1 unit of damage</p>
            <p>- <span class="grey-text">Grey sectors</span> are safe passages with no damage</p>
            <p>Your goal is to complete all missions while minimizing total radiation damage to your ship.</p>
        </div>
        
        <p class="start-text">Press any key to begin your mission, Explorer!</p>
    `,
    choices: "ALL_KEYS"
};

// New grid message
const newGridMessage = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return `
            <h2>New Asteroid Field Detected</h2>
            <p>Your ship has entered a new sector of space.</p>
            <p>Prepare for the next set of navigation decisions.</p>
            <p>Press any key to continue the mission.</p>
        `;
    },
    choices: "ALL_KEYS",
    on_finish: function() {
        grid.resetGrid(); // Reset the grid for the new set of trials
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
                <div class="choice-box blue-path" id="blue-choice" style="${lastTrialData.choice === 'blue' ? '' : 'visibility: hidden;'}">Route: <strong>Blue Path</strong></div>
                <div class="choice-box green-path" id="green-choice" style="${lastTrialData.choice === 'green' ? '' : 'visibility: hidden;'}">Route: <strong>Green Path</strong></div>
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

// End message
const end = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return `
            <h1>Mission Complete!</h1>
            <p>Congratulations, Space Explorer!</p>
            <p>You've successfully navigated all asteroid fields.</p>
            <p>Final Ship Damage: <strong>${totalCost} Units</strong></p>
            <p>Your exploration data has been recorded for analysis.</p>
            <p>Press any key to see your mission data.</p>
        `;
    },
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