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
        this.nTrials = gridData.env_costs.n_trials
        this.nGrids = gridData.env_costs.n_grids
        this.nCities = gridData.env_costs.n_cities
        this.observedCosts = {}; // Track observed costs for each grid
        this.currentGrid = 0; // Track the current grid
        this.currentCity = null; // Track the current city
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
    createGridHTML = function(trialIndex, selectedPath = null) {
        const trial = this.getTrialInfo(trialIndex);
        const city = trial.city;
        const grid = trial.grid;
        const binaryCosts = this.getBinaryCosts(`city_${city}_grid_${grid}`);
        const gridSize = this.gridSize;
        const jobNumber = (trialIndex % this.nTrials) + 1; // Job number within the grid
        
        let gridHTML = `
            <div class="current-job-container">
                <div class="cost-display-container">
                    <h2 class="cost-total">Total Tolls Paid:</h2>
                    <p id="total-cost" class="cost-total">$${totalCost}</p>
                    <p id="trial-cost" class="cost-trial hidden">-$0</p> 
                </div>
                <div class="grid-container" style="grid-template-columns: repeat(${gridSize}, 40px);">
        `;
    
        for (let row = 0; row < gridSize; row++) {
            for (let col = 0; col < gridSize; col++) {
                const cellId = `cell-${row}-${col}`;
                const isStartA = selectedPath !== 'green' && row === trial.start_A[0] && col === trial.start_A[1];
                const isStartB = selectedPath !== 'blue' && row === trial.start_B[0] && col === trial.start_B[1];
                const isGoalA = selectedPath !== 'green' && row === trial.goal_A[0] && col === trial.goal_A[1];
                const isGoalB = selectedPath !== 'blue' && row === trial.goal_B[0] && col === trial.goal_B[1];
                const isPathA = selectedPath !== 'green' && trial.path_A.some(coord => coord[0] === row && coord[1] === col);
                const isPathB = selectedPath !== 'blue' && trial.path_B.some(coord => coord[0] === row && coord[1] === col);
    
                const observedCost = this.observedCosts[`${row}-${col}`];
                const observedClass = observedCost !== undefined ? 
                    (observedCost === -1 ? 'observed-cost' : 'observed-no-cost') : '';
    
                if (isStartA) {
                    gridHTML += `<div class="grid-cell start blue-path ${observedClass}" id="${cellId}">S</div>`;
                } else if (isStartB) {
                    gridHTML += `<div class="grid-cell start green-path ${observedClass}" id="${cellId}">S</div>`;
                } else if (isGoalA) {
                    gridHTML += `<div class="grid-cell goal blue-path ${observedClass}" id="${cellId}">🏠</div>`;
                } else if (isGoalB) {
                    gridHTML += `<div class="grid-cell goal green-path ${observedClass}" id="${cellId}">🏠</div>`;
                } else if (isPathA || isPathB) {
                    const pathClass = isPathA ? 'blue-path' : 'green-path';
                    gridHTML += `<div class="grid-cell ${observedClass} ${pathClass}" id="${cellId}" style="font-size: 1.5rem;">⚝</div>`;
                } else {
                    gridHTML += `<div class="grid-cell ${observedClass}" id="${cellId}"></div>`;
                }
            }
        }   
        gridHTML += `</div></div>`;
    
        return gridHTML;
    };
    
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
            trialCostElement.textContent = "-$0";
            trialCostElement.classList.add("hidden");
        }
    
        // Reset total cost
        totalCost = 0;
        const totalCostElement = document.getElementById("total-cost");
        if (totalCostElement) {
            totalCostElement.textContent = "$0";
        }
    }
    
    // Check if the city has changed for the upcoming trial
    hasCityChanged(trialIndex) {
        const upcomingTrial = this.getTrialInfo(trialIndex);
        const upcomingCity = upcomingTrial.city;
        
        // If this is the first trial, set the current city and return false
        if (this.currentCity === null) {
            this.currentCity = upcomingCity;
            return false;
        }
        
        // Check if the city has changed
        if (upcomingCity !== this.currentCity) {
            this.currentCity = upcomingCity;
            return true;
        }
        
        return false;
    }
    
    // Get the current city
    getCurrentCity() {
        return this.currentCity;
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

// 1. Add taxi character with animation
function createAvatar() {
    return `
        <img src="assets/vehicles/taxi.png" width="30" height="30" alt="Taxi Avatar" />
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
                    cellElement.style.backgroundColor = cost === -1 ? "#f87171" : "#b8b8d9"; // Red for toll, grey for free
                }

                if (cost === -1) {
                    trialCost++;

                    // Visual burst effect for toll cost
                    cellElement.innerHTML += '<div class="cost-burst">+$1 Toll</div>';
                    setTimeout(() => {
                        const burst = cellElement.querySelector('.cost-burst');
                        if (burst) burst.remove();
                    }, 600);

                    // Play toll sound
                    const costSound = new Audio('assets/costSound.mp3');
                    costSound.play();

                    if (!trialCostVisible) {
                        const trialCostElement = document.getElementById("trial-cost");
                        if (trialCostElement) {
                            trialCostElement.classList.remove("hidden");
                            trialCostVisible = true;
                        }
                    }
                } else {
                    // Visual feedback for free passage
                    cellElement.innerHTML += '<div class="free-burst">Free</div>';
                    setTimeout(() => {
                        const burst = cellElement.querySelector('.free-burst');
                        if (burst) burst.remove();
                    }, 600);

                    // Play free passage sound
                    // const freeSound = new Audio('assets/freeSound.mp3');
                    // freeSound.play();
                }

                const trialCostElement = document.getElementById("trial-cost");
                if (trialCostElement) {
                    trialCostElement.textContent = `-$${trialCost}`;
                }

                // remove the star or S or G, then add the avatar
                cellElement.textContent = '';
                cellElement.classList.add('avatar');
                cellElement.innerHTML += createAvatar(); // Add taxi avatar on top
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
                totalCostElement.textContent = `$${currentCount}`;
                
                if (frame === totalFrames) {
                    clearInterval(counter);
                    totalCostElement.textContent = `$${totalCost}`;
                    
                    // Reset trial cost display with animation
                    trialCostElement.textContent = `-$0`;
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

// 5. Update the path selection trial to include taxi theme elements
const pathSelectionTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return `
            <div class="jobs-layout">
                <div class="current-job-section">
                    ${grid.createGridHTML(currentTrialIndex)}
                    <div class="choice-container">
                        <div class="choice-box blue-path" id="blue-choice">
                            <div class="choice-icon" style="font-size: 4rem; font-weight: bold;">←</div>
                        </div>
                        <div class="choice-box green-path" id="green-choice">
                            <div class="choice-icon" style="font-size: 4rem; font-weight: bold;">→</div>
                        </div>
                    </div>
                </div>
                ${createUpcomingJobsHTML(currentTrialIndex)}
            </div>
        `;
    },
    choices: ['arrowleft', 'arrowright'], 
    on_finish: function(data) {
        console.log("Key pressed:", data.response);
    
        let choice;
        if (data.response === 'arrowleft') {
            choice = 'blue';
        } else if (data.response === 'arrowright') {
            choice = 'green';
        } else {
            console.error("Invalid keypress:", data.response);
            return;
        }
    
        console.log("Chosen path:", choice);
    
        // Add "swipe" effect on selection
        const choiceElement = document.getElementById(`${choice}-choice`);
        const unchosenElement = document.getElementById(choice === 'blue' ? 'green-choice' : 'blue-choice');
        
        if (choiceElement && unchosenElement) {
            choiceElement.classList.add('choice-selected');
            unchosenElement.classList.add('choice-unselected');
        }

        // Replot the grid with only the chosen path
        const gridContainer = document.querySelector(".current-job-section");
        if (gridContainer) {
            gridContainer.innerHTML = grid.createGridHTML(currentTrialIndex, choice);
        }
        
        // Store the choice in the trial data
        data.choice = choice;
        jsPsych.data.get().addToLast({ choice: data.choice });
    }
};

// Add this function to create HTML for upcoming job previews
function createUpcomingJobsHTML(currentTrialIndex) {
    // Calculate which grid we're in (nTrials is contained in the grid object)
    const currentGridNumber = Math.floor(currentTrialIndex / grid.nTrials);
    const currentGridStartIndex = currentGridNumber * grid.nTrials;
    const currentGridEndIndex = currentGridStartIndex + grid.nTrials -1; // Last trial index in this grid
    
    // Only show jobs within the current grid
    const remainingTrialsInGrid = currentGridEndIndex - currentTrialIndex;
    
    if (remainingTrialsInGrid <= 0) {
        return ''; // No upcoming jobs in this grid
    }
    
    let upcomingHTML = `
        <div class="jobs-section">
            <h3 class="jobs-header">Upcoming Jobs</h3>
            <div class="upcoming-jobs-container">
    `;
    
    for (let i = 1; i <= remainingTrialsInGrid; i++) {
        const previewIndex = currentTrialIndex + i;
        const trial = grid.getTrialInfo(previewIndex);
        const city = trial.city;
        const gridId = trial.grid;
        const jobNumber = (previewIndex % grid.nTrials) + 1; // Job number within the grid
        
        upcomingHTML += `
            <div class="upcoming-job">
                <div class="upcoming-grid" style="grid-template-columns: repeat(${grid.gridSize}, 25px);">
        `;
        
        for (let row = 0; row < grid.gridSize; row++) {
            for (let col = 0; col < grid.gridSize; col++) {
                const isStartA = row === trial.start_A[0] && col === trial.start_A[1];
                const isStartB = row === trial.start_B[0] && col === trial.start_B[1];
                const isGoalA = row === trial.goal_A[0] && col === trial.goal_A[1];
                const isGoalB = row === trial.goal_B[0] && col === trial.goal_B[1];
                const isPathA = trial.path_A.some(coord => coord[0] === row && coord[1] === col);
                const isPathB = trial.path_B.some(coord => coord[0] === row && coord[1] === col);
                
                // Check if this cell has been observed in previous trials
                const observedCost = grid.observedCosts[`${row}-${col}`];
                const observedClass = observedCost !== undefined ? 
                    (observedCost === -1 ? 'observed-cost' : 'observed-no-cost') : '';
                
                if (isStartA) {
                    upcomingHTML += `<div class="upcoming-cell start blue-path ${observedClass}">S</div>`;
                } else if (isStartB) {
                    upcomingHTML += `<div class="upcoming-cell start green-path ${observedClass}">S</div>`;
                } else if (isGoalA) {
                    upcomingHTML += `<div class="upcoming-cell goal blue-path ${observedClass}">🏠</div>`;
                } else if (isGoalB) {
                    upcomingHTML += `<div class="upcoming-cell goal green-path ${observedClass}">🏠</div>`;
                } else if (isPathA) {
                    upcomingHTML += `<div class="upcoming-cell blue-path ${observedClass}">⚝</div>`;
                } else if (isPathB) {
                    upcomingHTML += `<div class="upcoming-cell green-path ${observedClass}">⚝</div>`;
                } else {
                    upcomingHTML += `<div class="upcoming-cell ${observedClass}"></div>`;
                }
            }
        }
        
        upcomingHTML += `
                </div>
            </div>
        `;
    }
    
    upcomingHTML += `
            </div>
        </div>
    `;
    return upcomingHTML;
}

// Ensure setCityBackground is correctly implemented
function setCityBackground(cityId) {
    const body = document.body;

    // Clear any existing background styles before applying a new one
    body.style.backgroundImage = '';
    body.style.backgroundSize = '';
    body.style.backgroundPosition = '';
    body.style.backgroundRepeat = '';

    // Apply the new background
    body.style.backgroundImage = `url('assets/cities/${cityId}.png')`;
    body.style.backgroundSize = 'cover';
    body.style.backgroundPosition = 'center';
    body.style.backgroundRepeat = 'no-repeat';

    console.log(`Set background to city${cityId}.png`);
}

// Modified newGridMessage to ensure city background updates correctly
const newGridMessage = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const nextTrialIndex = currentTrialIndex; // Next trial will be this index
        const isCityChanged = grid.hasCityChanged(nextTrialIndex);

        if (isCityChanged) {
            cityId = grid.getTrialInfo(nextTrialIndex).city;
            console.log("City changed to:", cityId);
            setCityBackground(cityId);
            return `
                <h2>New City!</h2>
                <p>Your taxi company is now operating in a new city.</p>
                <p>The streets and traffic patterns may be different here.</p>
                <p>Prepare for the next set of route decisions.</p>
                <p>Press any key to continue dispatching.</p>
            `;
        } else {
            return `
                <h2>New Day</h2>
                <p>A new day has begun, and the tolls in this city have been reset.</p>
                <p>Prepare for the next set of route decisions.</p>
                <p>Press any key to continue dispatching.</p>
            `;
        }
    },
    choices: "ALL_KEYS",
    on_finish: function() {
        grid.resetGrid(); // Reset the grid for the new set of trials
    }
};

const pathAnimationTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const lastTrialData = jsPsych.data.get().last(1).values()[0];
        
        return `
            <div class="jobs-layout">
                <div class="current-job-section">
                    ${grid.createGridHTML(currentTrialIndex, lastTrialData.choice)}
                </div>
                ${createUpcomingJobsHTML(currentTrialIndex)}
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
        const binaryCosts = grid.getBinaryCosts(`city_${currentTrial.city}_grid_${currentTrial.grid}`);

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
            <h1>Shift Complete!</h1>
            <p>Great job, Dispatcher!</p>
            <p>You've successfully completed all taxi assignments.</p>
            <p>Total Toll Costs: <strong>$${totalCost}</strong></p>
            <p>Your performance data has been recorded for evaluation.</p>
            <p>Press any key to see your dispatch summary.</p>
        `;
    },
    choices: "ALL_KEYS"
};

// Modified instructions
const instructions = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div class="instruction-section">
            <h1>Taxi Dispatch Coordinator</h1>
            <p>Welcome to City Cabs! As the dispatch coordinator, you must decide which taxi jobs to accept.</p>
        </div>
        
        <div class="instruction-section">
            <h2>Job Selection:</h2>
            <p>For each dispatch, you'll see two possible routes marked with stars:</p>
            <p>- <span class="blue-text">Blue stars</span> mark the first route</p>
            <p>- <span class="green-text">Green stars</span> mark the second route</p>
            <p>Each route has a pickup point (S) and a drop-off destination (G).</p>
        </div>
        
        <div class="instruction-section">
            <h2>Your Task:</h2>
            <p>Choose which route to assign to your taxi using your arrow keys:</p>
            <p>- Press <strong><span class="blue-text">LEFT ARROW</span></strong> to assign the blue route</p>
            <p>- Press <strong><span class="green-text">RIGHT ARROW</span></strong> to assign the green route</p>
        </div>
        
        <div class="instruction-section">
            <h2>Toll Roads:</h2>
            <p>Some streets contain toll roads that cost money to travel:</p>
            <p>- <span class="red-text">Red streets</span> are toll roads that cost $1 to pass through</p>
            <p>- <span class="grey-text">Grey streets</span> are free roads with no tolls</p>
            <p>Your goal is to complete all taxi jobs while minimizing total toll costs for your company.</p>
        </div>
        
        <p class="start-text">Press any key to begin your shift, Dispatcher!</p>
    `,
    choices: "ALL_KEYS",
    on_load: function() {
        // Set initial city background from the first trial
        // Set the city based on the first trial's city
        const firstTrial = grid.getTrialInfo(0);
        const cityId = firstTrial.city;
        grid.currentCity = cityId; // Initialize the current city
        setCityBackground(cityId);
    }
};

// Create timeline
function createTimeline() {
    const timeline = [instructions];

    // Loop through all trials and add them to the timeline
    for (let i = 0; i < grid.trialInfo.length; i++) {
        if (i % grid.nTrials === 0 && i !== 0) {
            // Add new grid message after each grid
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