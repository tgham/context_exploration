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
    createGridHTML = function(trialIndex, selectedPath = null, keyAssignment = null) {
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
    
                // Handle overlapping paths
                const isOverlap = isPathA && isPathB;
                let pathClass = '';
                let content = ''; // Content for the cell (e.g., letter or other marker)
                
                // Determine content based on key assignment if provided
                if (keyAssignment) {
                    if (isOverlap) {
                        const randomChoice = Math.random() < 0.5;
                        pathClass = randomChoice ? 'blue-path' : 'green-path';
                        content = randomChoice ? 
                            `<span class="green-text">${keyAssignment.green}</span>` : 
                            `<span class="blue-text">${keyAssignment.blue}</span>`;
                    } else if (isPathA) {
                        pathClass = 'blue-path';
                        content = keyAssignment.blue;
                    } else if (isPathB) {
                        pathClass = 'green-path';
                        content = keyAssignment.green;
                    }
                } else {
                    // Fall back to stars if no key assignment provided
                    if (isOverlap) {
                        const randomChoice = Math.random() < 0.5;
                        pathClass = randomChoice ? 'blue-path' : 'green-path';
                        content = randomChoice ? '<span class="green-text">⚝</span>' : '<span class="blue-text">⚝</span>';
                    } else if (isPathA) {
                        pathClass = 'blue-path';
                        content = '⚝';
                    } else if (isPathB) {
                        pathClass = 'green-path';
                        content = '⚝';
                    }
                }
    
                if (isStartA) {
                    gridHTML += `<div class="grid-cell start blue-path ${observedClass}" id="${cellId}">
                                    <img src="assets/people/blue_person.png" alt="Blue Start" width="30" height="30">
                                 </div>`;
                } else if (isStartB) {
                    gridHTML += `<div class="grid-cell start green-path ${observedClass}" id="${cellId}">
                                    <img src="assets/people/green_person.png" alt="Green Start" width="30" height="30">
                                 </div>`;
                } else if (isGoalA) {
                    gridHTML += `<div class="grid-cell goal blue-path ${observedClass}" id="${cellId}">🏠</div>`;
                } else if (isGoalB) {
                    gridHTML += `<div class="grid-cell goal green-path ${observedClass}" id="${cellId}">🏠</div>`;
                } else if (isPathA || isPathB || isOverlap) {
                    gridHTML += `<div class="grid-cell ${observedClass} ${pathClass}" id="${cellId}" style="font-size: 2rem;">${content}</div>`;
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
let cityMapping = {}; // This will store our shuffled mapping

fetch('assets/trial_sequences/expt_info_1.json')
    .then(response => response.json())
    .then(data => {
        grid = new Grid(data); // Initialize the Grid class with the loaded data
        
        // Create a shuffled mapping for cities
        createCityMapping(8); // Assuming you have 8 city files
        
        console.log('Grid data loaded:', grid);
        console.log('City mapping created:', cityMapping);
        
        initializeExperiment(); // Call a function to start the experiment
    })
    .catch(error => console.error('Error loading JSON:', error));

// Function to create a random mapping of city IDs
function createCityMapping(numCities) {
    // Create an array of city IDs (1 to numCities)
    let cityIds = Array.from({length: numCities}, (_, i) => i + 1);
    
    // Shuffle the array
    cityIds = shuffleArray(cityIds);
    
    // Create the mapping
    for (let i = 1; i <= numCities; i++) {
        cityMapping[i] = cityIds[i-1];
    }
}

// Fisher-Yates shuffle algorithm
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

// Function to animate the agent along the chosen path
let totalCost = 0; // Keeps track of total cost across trials

// 1. Add taxi character with animation
function createAvatar() {
    return `
        🚖
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
                }

                const trialCostElement = document.getElementById("trial-cost");
                if (trialCostElement) {
                    trialCostElement.textContent = `-$${trialCost}`;
                }

                // Update observed costs in upcoming grids
                const upcomingCells = document.querySelectorAll(`.upcoming-cell[data-row="${curRow}"][data-col="${curCol}"]`);
                upcomingCells.forEach(upcomingCell => {
                    upcomingCell.classList.remove("observed-cost", "observed-no-cost");
                    upcomingCell.classList.add(cost === -1 ? "observed-cost" : "observed-no-cost");
                });

                // Remove the star or S or G, then add the avatar
                cellElement.textContent = '';
                cellElement.classList.add('avatar');
                cellElement.innerHTML += createAvatar(); // Add taxi avatar on top
            } else {
                console.error(`Cell not found in DOM: cell-${curRow}-${curCol}`);
                return;
            }

            currentStep++;
            setTimeout(step, 400);
        } else {
            // Animation complete
            mergeCosts(trialCost, callback);
        }
    }

    setTimeout(step, 400);
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
            const frameDuration = 1000 / 60;
            const totalFrames = Math.round(duration / frameDuration);
            let frame = 0;
            
            const counter = setInterval(() => {
                frame++;
                const progress = frame / totalFrames;
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
        // Add transition effect to fade out the current job
        const currentJob = document.querySelector(".grid-container");
        if (currentJob) {
            currentJob.classList.add("fade-out");
        }

        // Add transition effect to fade out the leftmost upcoming job
        const upcomingJobs = document.querySelectorAll(".upcoming-job");
        if (upcomingJobs.length > 0) {
            const leftmostJob = upcomingJobs[0];
            leftmostJob.classList.add("fade-out");
        }

        setTimeout(() => {
            currentTrialIndex++;
            jsPsych.finishTrial();
            
            setTimeout(() => {
                // Remove fade-transition class after the transition
                if (currentJob) currentJob.classList.remove("fade-out");
                if (upcomingJobs.length > 0) {
                    const leftmostJob = upcomingJobs[0];
                    if (leftmostJob) leftmostJob.classList.remove("fade-out");
                }
            }, 400);
        }, 400);
    }, 1000);
}

// 5. Update the path selection trial to include taxi theme elements
const pathSelectionTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        // Randomly assign F and J to blue and green paths
        const keyAssignment = Math.random() < 0.5 ? 
            { blue: 'F', green: 'J' } : 
            { blue: 'J', green: 'F' };
        
        // Store the assignment for this trial
        jsPsych.data.addProperties({
            blue_key: keyAssignment.blue,
            green_key: keyAssignment.green
        });
        
        return `
            <div class="jobs-layout">
                <div class="current-job-section grid-fade-in">
                    ${grid.createGridHTML(currentTrialIndex, null, keyAssignment)}
                </div>
                <div class="upcoming-jobs-container grid-fade-in">
                    ${createUpcomingJobsHTML(currentTrialIndex)}
                </div>
            </div>
        `;
    },
    choices: ['f', 'j'], 
    on_finish: function(data) {
        // Get the key assignment for this trial
        const keyAssignment = {
            blue: jsPsych.data.get().last(1).values()[0].blue_key,
            green: jsPsych.data.get().last(1).values()[0].green_key
        };
        
        // Determine the chosen path based on the key pressed
        let choice;
        if (data.response === 'f') {
            choice = keyAssignment.blue === 'F' ? 'blue' : 'green';
        } else if (data.response === 'j') {
            choice = keyAssignment.blue === 'J' ? 'blue' : 'green';
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
            gridContainer.innerHTML = grid.createGridHTML(currentTrialIndex, choice, keyAssignment);
        }
        
        // Store all the relevant data from the current trial
        const currentTrial = grid.getTrialInfo(currentTrialIndex);
        data.choice = choice;
        data.trial = currentTrial.trial;
        data.city = currentTrial.city;
        data.grid_id = currentTrial.grid;
        data.path_chosen = choice;
        data.button_pressed = data.response;
        data.reaction_time_ms = data.rt;
        data.key_assignment = keyAssignment;

        // Include all columns from the current trial
        Object.keys(currentTrial).forEach(key => {
            data[key] = currentTrial[key];
        });

        // Include all trial info from the current trial
        Object.assign(data, currentTrial);
        
        // Add the trial data to jsPsych's data collection
        jsPsych.data.get().addToLast(data);
    }
};

// Add this function to create HTML for upcoming job previews
function createUpcomingJobsHTML(currentTrialIndex) {
    // Calculate which grid we're in (nTrials is contained in the grid object)
    const currentGridNumber = Math.floor(currentTrialIndex / grid.nTrials);
    const currentGridStartIndex = currentGridNumber * grid.nTrials;
    const currentGridEndIndex = currentGridStartIndex + grid.nTrials - 1; // Last trial index in this grid

    // Only show jobs within the current grid
    const remainingTrialsInGrid = currentGridEndIndex - currentTrialIndex;

    if (remainingTrialsInGrid <= 0) {
        return ''; // No upcoming jobs in this grid
    }

    let upcomingHTML = `
        <div class="jobs-section">
            <div class="upcoming-jobs-header-container">
                <div class="upcoming-jobs-header">Upcoming jobs</div>
            </div>
            <div class="upcoming-jobs-mask-container">
                <div class="upcoming-jobs-actual-container">
    `;

    for (let i = 1; i <= remainingTrialsInGrid; i++) {
        const previewIndex = currentTrialIndex + i;
        const trial = grid.getTrialInfo(previewIndex);
        const jobNumber = (previewIndex % grid.nTrials) + 1; // Job number within the grid

        upcomingHTML += `
            <div class="upcoming-job">
                <div class="upcoming-grid" style="grid-template-columns: repeat(${grid.gridSize}, 30px); grid-auto-rows: 30px;">
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

                // Handle overlapping paths
                const isOverlap = isPathA && isPathB;
                let pathClass = '';
                let content = ''; // Content for the cell (e.g., star or other marker)
                if (isOverlap) {
                    const randomChoice = Math.random() < 0.5;
                    pathClass = randomChoice ? 'blue-path' : 'green-path';
                    content = randomChoice ? '<span class="green-text">⚝</span>' : '<span class="blue-text">⚝</span>';
                } else if (isPathA) {
                    pathClass = 'blue-path';
                    content = '⚝';
                } else if (isPathB) {
                    pathClass = 'green-path';
                    content = '⚝';
                }

                if (isStartA) {
                    upcomingHTML += `<div class="upcoming-cell ${observedClass} blue-path" data-row="${row}" data-col="${col}">
                                        <img src="assets/people/blue_person.png" alt="Blue Start" width="20" height="20">
                                     </div>`;
                } else if (isStartB) {
                    upcomingHTML += `<div class="upcoming-cell ${observedClass} green-path" data-row="${row}" data-col="${col}">
                                        <img src="assets/people/green_person.png" alt="Green Start" width="20" height="20">
                                     </div>`;
                } else if (isGoalA || isGoalB || isPathA || isPathB || isOverlap) {
                    upcomingHTML += `<div class="upcoming-cell ${observedClass} ${pathClass}" data-row="${row}" data-col="${col}" style="font-size: 1.5rem;">
                                        ${isGoalA || isGoalB ? '🏠' : content}
                                     </div>`;
                } else {
                    upcomingHTML += `<div class="upcoming-cell ${observedClass}" data-row="${row}" data-col="${col}"></div>`;
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
        </div>
    `;
    return upcomingHTML;
}

// Update the setCityBackground function to use the mapping
function setCityBackground(cityId) {
    const body = document.body;
    
    // Use the mapped city ID if cityId is not 'practice1' or 'practice2'
    const mappedCityId = (cityId === 'practice1' || cityId === 'practice2') ? cityId : cityMapping[cityId];

    // Clear any existing background styles before applying a new one
    body.style.backgroundImage = '';
    body.style.backgroundSize = '';
    body.style.backgroundPosition = '';
    body.style.backgroundRepeat = '';

    // Apply the new background
    body.style.backgroundImage = `url('assets/cities/${mappedCityId}.png')`;
    body.style.backgroundSize = 'cover';
    body.style.backgroundPosition = 'center';
    body.style.backgroundRepeat = 'no-repeat';
}

// Modified function structure: separate functions for city change vs day change
function animateCityChange(oldCityId, newCityId) {
    console.log(`City Change Animation: from ${oldCityId} to ${newCityId}`);
    
    // Create a container for the animation
    let transitionContainer = document.createElement('div');
    transitionContainer.style.position = 'fixed';
    transitionContainer.style.top = '0';
    transitionContainer.style.left = '0';
    transitionContainer.style.width = '200%'; // Double width to fit both images
    transitionContainer.style.height = '100%';
    transitionContainer.style.zIndex = '1000';
    transitionContainer.style.display = 'flex';
    transitionContainer.style.transition = 'transform 1.5s ease-in-out';
    document.body.appendChild(transitionContainer);
    
    // Create old city element
    let oldCity = document.createElement('div');
    oldCity.style.width = '50%'; // Half of the container
    oldCity.style.height = '100%';
    oldCityMapping = cityMapping[oldCityId];
    oldCity.style.backgroundImage = `url('assets/cities/${oldCityMapping}.png')`;
    oldCity.style.backgroundSize = 'cover';
    oldCity.style.backgroundPosition = 'center';
    transitionContainer.appendChild(oldCity);
    
    // Create new city element
    let newCity = document.createElement('div');
    newCity.style.width = '50%'; // Half of the container
    newCity.style.height = '100%';
    newCityMapping = cityMapping[newCityId];
    newCity.style.backgroundImage = `url('assets/cities/${newCityMapping}.png')`;
    newCity.style.backgroundSize = 'cover';
    newCity.style.backgroundPosition = 'center';
    transitionContainer.appendChild(newCity);
    
    // Force browser reflow before starting animation
    void transitionContainer.offsetWidth;
    
    // Start the slide animation
    transitionContainer.style.transform = 'translateX(-50%)';
    
    // After animation completes, set the new background and remove the container
    setTimeout(() => {
        setCityBackground(newCityId);
        document.body.removeChild(transitionContainer);
    }, 1600);
}

function animateDayChange(cityId) {
    console.log(`Day Change Animation: Staying in city ${cityId}`);
    const blackCover = document.createElement('div');
    blackCover.style.position = 'fixed';
    blackCover.style.top = '0';
    blackCover.style.left = '0';
    blackCover.style.width = '100%';
    blackCover.style.height = '100%';
    blackCover.style.backgroundColor = 'black';
    blackCover.style.opacity = '0';
    blackCover.style.transition = 'opacity 1s ease-in-out';
    blackCover.style.zIndex = '1000';
    document.body.appendChild(blackCover);

    // Fade to full opacity
    setTimeout(() => {
        blackCover.style.opacity = '0.6';
    }, 10);

    // After 1s, set the new city background and fade back to transparency
    setTimeout(() => {
        setCityBackground(cityId);
        blackCover.style.opacity = '0';
    }, 1000);

    // Remove the black cover after the transition is complete
    setTimeout(() => {
        document.body.removeChild(blackCover);
    }, 2000);
}

// display feedback after a grid
const gridFeedback = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const todayTolls = totalCost; // Assuming totalCost tracks the tolls paid so far
        return `
            <div class="new-day-text">
                <h3>You paid a total of <strong style="color: red;">$${todayTolls}</strong> in tolls today.</h3>
                <h3>Press any key to continue.</h3>
            </div>
        `;
    },
    choices: "ALL_KEYS",
    on_finish: function() {
        grid.resetGrid(); // Reset the grid for the next set of trials
    }
};


const newGridMessage = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const nextTrialIndex = currentTrialIndex; // Next trial will be this index
        const nextTrial = grid.getTrialInfo(nextTrialIndex);
        const nextCityId = nextTrial.city;
        const currentCityId = grid.getCurrentCity();
        let message;

        console.log("Current city:", currentCityId);
        console.log("Next city:", nextCityId);
        
        // Explicitly check if city has changed by comparing IDs
        if (currentCityId !== null && nextCityId !== currentCityId) {
            console.log("City changed from", currentCityId, "to", nextCityId);
            // Run slide animation
            animateCityChange(currentCityId, nextCityId);
            
            message = `
                <div class="new-day-text">
                    <h2>New City!</h2>
                    <p>Your taxi company is now operating in a new city.</p>
                    <p>The streets and traffic patterns may be different here.</p>
                    <p>Prepare for the next set of route decisions.</p>
                    <p id="continue-text" style="display: none;">Press any key to continue dispatching.</p>
                </div>
            `;
            
            // Update the current city
            grid.currentCity = nextCityId;
        } else {
            console.log("Same city - just a new day");
            // Run fade animation for new day in same city
            animateDayChange(currentCityId || nextCityId);
            
            const todayTolls = totalCost; // Assuming totalCost tracks the tolls paid so far
            message = `
                <div class="new-day-text">
                    <h2>New Day</h2>
                    <p>A new day has begun, and the tolls in this city have been reset.</p>
                    <p>Prepare for the next set of route decisions.</p>
                    <p id="continue-text" style="display: none;">Press any key to continue dispatching.</p>
                </div>
            `;
            
            // Ensure city is set if this is the first trial
            if (currentCityId === null) {
                grid.currentCity = nextCityId;
            }
        }

        // After 2s, show the text and enable keypresses manually
        setTimeout(() => {
            document.getElementById("continue-text").style.display = "block";
            
            // Manually register keypress listener
            jsPsych.pluginAPI.getKeyboardResponse({
                callback_function: jsPsych.finishTrial, // Ends trial when a key is pressed
                valid_responses: "ALL_KEYS",
                rt_method: "performance",
                persist: false,
                allow_held_key: false
            });
        }, 2500); // Increased delay to allow animation to complete

        return message;
    },
    choices: "NO_KEYS", // Initially disable keypresses
    on_finish: function() {
        grid.resetGrid(); // Reset the grid for the new set of trials
    }
};

const firstGridMessage = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {

        // Set initial city background from the first trial
        const firstTrial = grid.getTrialInfo(0);
        const cityId = firstTrial.city;
        grid.currentCity = cityId; // Initialize the current city
        setCityBackground(cityId); // Plot the first city background

        return `
            <div class="new-day-text">
                <h2>Welcome to the Experiment!</h2>
                <p>Your taxi company is starting operations</p>
                <p>Make the best route decisions to minimize toll costs.</p>
                <p>Press any key to begin dispatching.</p>
            </div>
        `;
    },
    choices: "ALL_KEYS",
    on_finish: function() {
        console.log("Experiment has begun in City:", grid.getCurrentCity());
    }
};





const pathAnimationTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const lastTrialData = jsPsych.data.get().last(1).values()[0];
        const keyAssignment = lastTrialData.key_assignment;
        
        return `
            <div class="jobs-layout">
                <div class="current-job-section">
                    ${grid.createGridHTML(currentTrialIndex, lastTrialData.choice, keyAssignment)}
                </div>
                <div class="upcoming-jobs-container">
                    ${createUpcomingJobsHTML(currentTrialIndex)}
                </div>
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
            <div class="button-container">
                <button id="download-data" class="download-button">Download Data</button>
                <p>Press any key to finish the experiment.</p>
            </div>
        `;
    },
    choices: "ALL_KEYS",
    on_load: function() {
        // Add event listener for the download button
        document.getElementById('download-data').addEventListener('click', downloadTrialData);
        
        // Also create a CSV version of the data with jsPsych's built-in function
        const csvData = jsPsych.data.get().filter({choice: ['blue', 'green']}).csv();
        jsPsych.data.addProperties({
            exported_data: csvData
        });
    }
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
            <p>Each route has a passenger <img src="assets/people/blue_person.png" alt="Blue Passenger" width="20" height="20"> or <img src="assets/people/green_person.png" alt="Green Passenger" width="20" height="20"> at a pickup point, and a drop-off destination 🏠.</p>
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
            <p>- <strong><span class="red-text">Red streets</span></strong> are toll roads that cost $1 to pass through</p>
            <p>- <strong><span style="color:rgb(194, 194, 229);">Light grey streets</span></strong> are free roads with no tolls</p>
            <p>- <strong><span style="color: rgb(114, 114, 150);">Dark grey streets</span></strong> are roads that haven't been visited yet</p>
            <p>Your goal is to complete all taxi jobs while minimizing total toll costs for your company.</p>
        </div>
        
        <div class="instruction-section">
            <h2>Press any key to begin your shift, Dispatcher!</h2>
        </div>
    `,
    choices: "ALL_KEYS",
    on_load: function() {
        // Set initial city background to 'practice1.png'
        setCityBackground('practice1');
        grid.currentCity = 'practice1'; // Initialize the current city
    }
};

// Create timeline
function createTimeline() {
    const timeline = [instructions];

    // Add the first grid message
    timeline.push(firstGridMessage);

    // Loop through all trials and add them to the timeline
    for (let i = 0; i < grid.trialInfo.length; i++) {
        if (i % grid.nTrials === 0 && i !== 0) {
            // Add new grid message after each grid
            timeline.push(gridFeedback);
            timeline.push(newGridMessage);
        }
        timeline.push(pathSelectionTrial);
        timeline.push(pathAnimationTrial); 
    }

    // Add the end message
    timeline.push(end);

    return timeline;
}

function downloadTrialData() {
    // Get all path selection trial data
    const pathData = jsPsych.data.get().filter({trial_type: 'html-keyboard-response'}).values();
    
    // Format the data for CSV
    const trialData = pathData.map(trial => {
        // Only include trials where a choice was made
        if (trial.choice) {
            if (trial.choice) {
                return trial; // Return the entire trial object
            }
        }
        return null;
    }).filter(item => item !== null);

    // Convert the data to CSV format
    const csvHeaders = "trial,city,grid_id,path_chosen,button_pressed,reaction_time_ms,context,grid\n";
    const csvRows = trialData.map(trial => 
        `${trial.trial},${trial.city},${trial.grid_id},${trial.path_chosen},${trial.button_pressed},${trial.reaction_time_ms},${trial.context},${trial.grid}`
    ).join("\n");
    const csvContent = csvHeaders + csvRows;

    // Create a Blob with the CSV data
    const dataBlob = new Blob([csvContent], {type: 'text/csv'});
    
    // Create a download link and trigger it
    const url = URL.createObjectURL(dataBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'path_selection_data.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Start experiment when the page loads
function initializeExperiment() {
    const timeline = createTimeline();
    jsPsych.run(timeline);
}