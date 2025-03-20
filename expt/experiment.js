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
        this.gridSize = gridData.env_costs.grid_size; // Grid size (N)
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
        const gridSize = this.gridSize;
        
        let gridHTML = `
            <div class="cost-display-container">
                <h2 class="cost-total">Total Ship Damage:</h2>
                <p id="total-cost" class="cost-total">${totalCost} Units</p>
                <p id="trial-cost" class="cost-trial hidden">+0 Units</p> 
            </div>
            <div class="grid-container" style="grid-template-columns: repeat(${gridSize}, 40px);">
        `;
    
        for (let row = 0; row < gridSize; row++) {
            for (let col = 0; col < gridSize; col++) {
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
            if (row < 0 || row > this.gridSize - 1 || col < 0 || col > this.gridSize - 1) {
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
        <img src="assets/ships/ship_black.png" width="30" height="30" alt="Spaceship Avatar" />
    `;
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
                prevCellElement.innerHTML = ''; // Remove avatar
            }
        }

        if (currentStep < path.length) {
            const [curRow, curCol] = path[currentStep];
            const cellElement = document.getElementById(`cell-${curRow}-${curCol}`);

            if (cellElement) {
                const cost = binaryCosts[curRow][curCol];

                // Update observed cost classes
                cellElement.classList.remove("observed-cost", "observed-no-cost");
                cellElement.classList.add(cost === -1 ? "observed-cost" : "observed-no-cost");

                // Ensure start and goal cells update their color when observed
                if (cellElement.classList.contains("start") || cellElement.classList.contains("goal")) {
                    cellElement.style.backgroundColor = cost === -1 ? "#f87171" : "#b8b8d9"; // Red for high-cost, grey for low-cost
                }

                if (cost === -1) {
                    trialCost++;

                    // Visual burst effect for radiation damage
                    cellElement.innerHTML += '<div class="cost-burst">+1 Damage</div>';
                    setTimeout(() => {
                        const burst = cellElement.querySelector('.cost-burst');
                        if (burst) burst.remove();
                    }, 600);

                    // Play radiation sound
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
                    // Visual feedback for safe passage
                    cellElement.innerHTML += '<div class="free-burst">Safe</div>';
                    setTimeout(() => {
                        const burst = cellElement.querySelector('.free-burst');
                        if (burst) burst.remove();
                    }, 600);

                    // Play safe passage sound
                    // const safeSound = new Audio('assets/safeSound.mp3');
                    // safeSound.play();
                }

                const trialCostElement = document.getElementById("trial-cost");
                if (trialCostElement) {
                    trialCostElement.textContent = `+${trialCost} Units`;
                }

                // remove the star or S or G, then add the avatar
                cellElement.textContent = '';
                cellElement.classList.add('avatar');
                cellElement.innerHTML += createAvatar(); // Add spaceship avatar on top
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
    choices: ['arrowleft', 'arrowright'], 
    on_finish: function(data) {
        console.log("Key pressed:", data.response); // Log the keypress
    
        let choice;
        if (data.response === 'arrowleft') {
            choice = 'blue';
        } else if (data.response === 'arrowright') {
            choice = 'green';
        } else {
            console.error("Invalid keypress:", data.response);
            return;
        }
    
        console.log("Chosen path:", choice); // Log the chosen path
    
        // Add "swipe" effect on selection
        const choiceElement = document.getElementById(`${choice}-choice`);
        const unchosenElement = document.getElementById(choice === 'blue' ? 'green-choice' : 'blue-choice');
        
        if (choiceElement && unchosenElement) {
            choiceElement.classList.add('choice-selected');
            unchosenElement.classList.add('choice-unselected');
        }
        
        // Store the choice in the trial data
        data.choice = choice;
        jsPsych.data.get().addToLast({ choice: data.choice });
    }
};

// Modified newGridMessage
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
    on_load: function() {
        // Set a new random planet background when entering a new grid
        setRandomPlanetBackground();
    },
    on_finish: function() {
        grid.resetGrid(); // Reset the grid for the new set of trials
    }
};

// Modified instructions
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
    choices: "ALL_KEYS",
    on_load: function() {
        // Set the initial planet background
        setRandomPlanetBackground();
    }
};

// Add this function to select a random planet background
function setRandomPlanetBackground() {
    // Generate a random number between 1 and 9
    const planetNum = Math.floor(Math.random() * 9) + 1;
    // Format the number with leading zero if needed
    const planetId = planetNum.toString().padStart(2, '0');
    // Set the background image
    document.body.style.backgroundImage = `url('assets/planets/planet${planetId}.png')`;
    document.body.style.backgroundSize = 'cover';
    document.body.style.backgroundPosition = 'center';
    document.body.style.backgroundRepeat = 'no-repeat';
    
    // Add a subtle overlay to ensure grid visibility
    // const overlay = document.getElementById('background-overlay');
    // if (!overlay) {
    //     const newOverlay = document.createElement('div');
    //     newOverlay.id = 'background-overlay';
    //     newOverlay.style.position = 'fixed';
    //     newOverlay.style.top = '0';
    //     newOverlay.style.left = '0';
    //     newOverlay.style.width = '100%';
    //     newOverlay.style.height = '100%';
    //     newOverlay.style.backgroundColor = 'rgba(12, 12, 29, 0.0)';
    //     newOverlay.style.zIndex = '-1';
    //     document.body.appendChild(newOverlay);
    // }
    
    console.log(`Set background to planet${planetId}.png`);
}


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
    // Set the initial planet background
    setRandomPlanetBackground();
    
    const timeline = createTimeline();
    jsPsych.run(timeline);
}