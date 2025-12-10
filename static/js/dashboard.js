// Global variables
let todayPieChart = null;
let weeklyTrendChart = null;
let refreshCountdown = 30;
let refreshInterval = null;
let countdownInterval = null;

// Utility function: Format duration
function formatDuration(seconds) {
    if (seconds < 60) {
        return `${Math.floor(seconds)}s`;
    } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        return `${minutes}m`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        if (minutes > 0) {
            return `${hours}h ${minutes}m`;
        } else {
            return `${hours}h`;
        }
    }
}

// Utility function: Format hours
function formatHours(seconds) {
    const hours = seconds / 3600;
    return `${hours.toFixed(1)}h`;
}

// Update current session information
async function updateCurrentSession() {
    try {
        const response = await fetch('/api/stats/current');

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        const content = document.getElementById('currentSessionContent');

        if (!content) {
            console.error('Element currentSessionContent not found');
            return;
        }

        if (result.success && result.has_session && result.data) {
            const session = result.data;

            // Validate required fields
            const state = session.state || 'unknown';
            const duration = session.duration || 0;
            const zone = session.zone || null;

            let sessionHTML = `
                <div class="session-info">
                    <div class="session-item">
                        <h4>Current State</h4>
                        <p>${translateState(state)}</p>
                    </div>
                    <div class="session-item">
                        <h4>Duration</h4>
                        <p>${formatDuration(duration)}</p>
                    </div>`;

            if (zone) {
                sessionHTML += `
                    <div class="session-item">
                        <h4>Zone</h4>
                        <p>${zone}</p>
                    </div>`;
            }

            sessionHTML += `</div>`;
            content.innerHTML = sessionHTML;
        } else {
            content.innerHTML = `
                <div class="session-info">
                    <div class="session-item">
                        <h4>Current State</h4>
                        <p>No activity detected</p>
                    </div>
                </div>
            `;
        }
    } catch (error) {
        console.error('Failed to fetch current session:', error);
        const content = document.getElementById('currentSessionContent');
        if (content) {
            content.innerHTML = `
                <div class="session-info">
                    <div class="session-item">
                        <h4>Error</h4>
                        <p style="color: #B54708;">Unable to load current status</p>
                    </div>
                </div>
            `;
        }
    }
}

// Translate state
function translateState(state) {
    const stateMap = {
        'sitting': '🪑 Sitting',
        'standing': '🧍 Standing',
        'lying': '🛏️ Lying',
        'sleeping': '😴 Sleeping',
        'unknown': '❓ Unknown'
    };
    return stateMap[state] || state;
}

// Update today's statistics
async function updateTodayStats() {
    try {
        const response = await fetch('/api/stats/today');
        const result = await response.json();

        if (result.success && result.data) {
            const stats = result.data;

            // Safely get data with defaults
            const sittingDuration = stats.sitting_duration || 0;
            const standingDuration = stats.standing_duration || 0;
            const lyingDuration = stats.lying_duration || 0;
            const totalSessions = stats.total_sessions || 0;

            const sittingEl = document.getElementById('todaySitting');
            const standingEl = document.getElementById('todayStanding');
            const lyingEl = document.getElementById('todayLying');
            const sessionsEl = document.getElementById('todaySessions');

            if (sittingEl) sittingEl.textContent = formatHours(sittingDuration);
            if (standingEl) standingEl.textContent = formatHours(standingDuration);
            if (lyingEl) lyingEl.textContent = formatHours(lyingDuration);
            if (sessionsEl) sessionsEl.textContent = totalSessions;

            // Update pie chart
            updateTodayPieChart(stats);
        }
    } catch (error) {
        console.error('Failed to fetch today\'s stats:', error);
        // Display error state
        const elements = ['todaySitting', 'todayStanding', 'todayLying', 'todaySessions'];
        elements.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = '-';
        });
    }
}

// Update sitting details statistics
async function updateSittingDetails() {
    try {
        const response = await fetch('/api/stats/sitting');
        const result = await response.json();

        const content = document.getElementById('sittingDetailsContent');

        if (!content) {
            console.error('Element sittingDetailsContent not found');
            return;
        }

        if (result.success && result.data) {
            const stats = result.data;

            // Safely get data with defaults
            const totalHours = (stats.total_duration_hours || 0).toFixed(1);
            const sessionCount = stats.session_count || 0;
            const avgDuration = stats.average_session_duration || 0;
            const longestSession = stats.longest_session || 0;

            content.innerHTML = `
                <div class="sitting-details-grid">
                    <div class="detail-item">
                        <h4>Total Duration</h4>
                        <p>${totalHours}h</p>
                    </div>
                    <div class="detail-item">
                        <h4>Sessions</h4>
                        <p>${sessionCount}</p>
                    </div>
                    <div class="detail-item">
                        <h4>Average Duration</h4>
                        <p>${formatDuration(avgDuration)}</p>
                    </div>
                    <div class="detail-item">
                        <h4>Longest Session</h4>
                        <p>${formatDuration(longestSession)}</p>
                    </div>
                </div>
            `;
        } else {
            content.innerHTML = `<p style="text-align: center; color: #938872;">No data available</p>`;
        }
    } catch (error) {
        console.error('Failed to fetch sitting stats:', error);
        const content = document.getElementById('sittingDetailsContent');
        if (content) {
            content.innerHTML = `<p style="text-align: center; color: #B54708;">Loading failed</p>`;
        }
    }
}

// Check prolonged sitting alert
async function checkProlongedSitting() {
    try {
        const response = await fetch('/api/alert/prolonged_sitting?threshold=30');
        const result = await response.json();

        const alert = document.getElementById('prolongedSittingAlert');
        const message = document.getElementById('prolongedSittingMessage');

        if (result.success && result.data.alert) {
            const duration = Math.floor(result.data.current_duration_minutes);
            message.textContent = `You've been sitting for ${duration} minutes. Time to move!`;
            alert.style.display = 'flex';
        } else {
            alert.style.display = 'none';
        }
    } catch (error) {
        console.error('Failed to check prolonged sitting:', error);
    }
}

// Create/update today's activity pie chart
function updateTodayPieChart(stats) {
    const ctx = document.getElementById('todayPieChart');

    if (todayPieChart) {
        todayPieChart.destroy();
    }

    todayPieChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Sitting', 'Standing', 'Lying'],
            datasets: [{
                label: 'Activity Duration (hours)',
                data: [
                    stats.sitting_duration / 3600,
                    stats.standing_duration / 3600,
                    stats.lying_duration / 3600
                ],
                backgroundColor: [
                    'rgba(181, 132, 94, 0.8)',
                    'rgba(107, 100, 86, 0.8)',
                    'rgba(147, 136, 114, 0.8)'
                ],
                borderColor: [
                    'rgba(181, 132, 94, 1)',
                    'rgba(107, 100, 86, 1)',
                    'rgba(147, 136, 114, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: {
                            size: 14
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Today\'s Activity Distribution'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.label || '';
                            if (label) {
                                label += ': ';
                            }
                            label += context.parsed.toFixed(1) + 'h';
                            return label;
                        }
                    }
                }
            }
        }
    });
}

// Update weekly trend chart
async function updateWeeklyTrendChart() {
    try {
        const response = await fetch('/api/stats/weekly');
        const result = await response.json();

        if (!result.success) {
            return;
        }

        const stats = result.data;
        const ctx = document.getElementById('weeklyTrendChart');

        if (weeklyTrendChart) {
            weeklyTrendChart.destroy();
        }

        // Prepare data
        const labels = stats.daily_breakdown.map(day => day.date);
        const sittingData = stats.daily_breakdown.map(day => (day.sitting / 3600).toFixed(1));
        const standingData = stats.daily_breakdown.map(day => (day.standing / 3600).toFixed(1));
        const lyingData = stats.daily_breakdown.map(day => (day.lying / 3600).toFixed(1));

        weeklyTrendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Sitting',
                        data: sittingData,
                        borderColor: 'rgb(181, 132, 94)',
                        backgroundColor: 'rgba(181, 132, 94, 0.2)',
                        tension: 0.3,
                        fill: true
                    },
                    {
                        label: 'Standing',
                        data: standingData,
                        borderColor: 'rgb(107, 100, 86)',
                        backgroundColor: 'rgba(107, 100, 86, 0.2)',
                        tension: 0.3,
                        fill: true
                    },
                    {
                        label: 'Lying',
                        data: lyingData,
                        borderColor: 'rgb(147, 136, 114)',
                        backgroundColor: 'rgba(147, 136, 114, 0.2)',
                        tension: 0.3,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: {
                                size: 14
                            }
                        }
                    },
                    title: {
                        display: true,
                        text: 'Weekly Activity Trend (hours)'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                label += context.parsed.y + 'h';
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Hours'
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Failed to update weekly trend chart:', error);
    }
}

// Update last update time
function updateLastUpdateTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US');
    document.getElementById('lastUpdate').textContent = timeString;
}

// Start refresh countdown
function startRefreshCountdown() {
    if (countdownInterval) {
        clearInterval(countdownInterval);
    }

    refreshCountdown = 30;
    document.getElementById('refreshCountdown').textContent = refreshCountdown;

    countdownInterval = setInterval(() => {
        refreshCountdown--;
        document.getElementById('refreshCountdown').textContent = refreshCountdown;

        if (refreshCountdown <= 0) {
            refreshCountdown = 30;
        }
    }, 1000);
}

// Update prediction information
async function updatePredictions() {
    try {
        // Get next sitting prediction
        const nextSittingResponse = await fetch('/api/prediction/next_sitting');
        const nextSittingResult = await nextSittingResponse.json();

        // Get optimal reminder time
        const reminderResponse = await fetch('/api/prediction/optimal_reminder');
        const reminderResult = await reminderResponse.json();

        const content = document.getElementById('predictionContent');

        if (nextSittingResult.success && reminderResult.success) {
            const prediction = nextSittingResult.data;
            const reminder = reminderResult.data;

            content.innerHTML = `
                <div class="prediction-grid">
                    <div class="prediction-item">
                        <h4>🎯 Next Session Prediction</h4>
                        <p class="prediction-value">${prediction.predicted_duration_minutes || 0}m</p>
                        <p class="prediction-confidence">Confidence: ${(prediction.confidence * 100).toFixed(0)}%</p>
                        <p class="prediction-note">${prediction.recommendation}</p>
                    </div>
                    <div class="prediction-item">
                        <h4>⏰ Recommended Reminder</h4>
                        <p class="prediction-value">${reminder.recommended_reminder_interval}m</p>
                        <p class="prediction-note">${reminder.pattern_description}</p>
                    </div>
                </div>
            `;
        } else {
            content.innerHTML = `<p>📊 Predictions will be available after collecting more data...</p>`;
        }
    } catch (error) {
        console.error('Failed to fetch predictions:', error);
        document.getElementById('predictionContent').innerHTML =
            `<p>Unable to load prediction data</p>`;
    }
}

// Update anomaly detection
async function updateAnomalyDetection() {
    try {
        const response = await fetch('/api/prediction/anomaly');
        const result = await response.json();

        const content = document.getElementById('anomalyContent');

        if (result.success && !result.data.error) {
            const anomaly = result.data;

            let severityClass = 'normal';
            let severityIcon = '✅';
            if (anomaly.is_anomaly) {
                if (anomaly.severity === 'high') {
                    severityClass = 'high';
                    severityIcon = '🔴';
                } else if (anomaly.severity === 'medium') {
                    severityClass = 'medium';
                    severityIcon = '🟡';
                }
            }

            content.innerHTML = `
                <div class="anomaly-result ${severityClass}">
                    <div class="anomaly-header">
                        <span class="anomaly-icon">${severityIcon}</span>
                        <span class="anomaly-message">${anomaly.message}</span>
                    </div>
                    <div class="anomaly-stats">
                        <div class="anomaly-stat-item">
                            <span class="label">Today's Sitting:</span>
                            <span class="value">${anomaly.today_sitting_hours}h</span>
                        </div>
                        <div class="anomaly-stat-item">
                            <span class="label">Average:</span>
                            <span class="value">${anomaly.average_sitting_hours}h</span>
                        </div>
                        <div class="anomaly-stat-item">
                            <span class="label">Deviation:</span>
                            <span class="value ${anomaly.deviation_percentage > 0 ? 'negative' : 'positive'}">
                                ${anomaly.deviation_percentage > 0 ? '+' : ''}${anomaly.deviation_percentage}%
                            </span>
                        </div>
                    </div>
                </div>
            `;
        } else {
            content.innerHTML = `<p>Insufficient data for anomaly detection</p>`;
        }
    } catch (error) {
        console.error('Failed to fetch anomaly detection:', error);
    }
}

// Update behavior prediction
async function updateBehaviorPrediction() {
    try {
        // Get current state
        const currentResponse = await fetch('/api/stats/current');
        const currentResult = await currentResponse.json();

        let currentState = 'unknown';
        if (currentResult.success && currentResult.has_session && currentResult.data && currentResult.data.state) {
            currentState = currentResult.data.state;
        }

        // Get prediction
        const predictionResponse = await fetch('/api/behavior/predict_current_state');
        const predictionResult = await predictionResponse.json();

        // Get suggestion
        const suggestionResponse = await fetch(`/api/behavior/smart_suggestion?current_state=${currentState}`);
        const suggestionResult = await suggestionResponse.json();

        const content = document.getElementById('behaviorPredictionContent');

        if (predictionResult.success && suggestionResult.success) {
            const prediction = predictionResult.data;
            const suggestion = suggestionResult.data;

            const stateIcons = {
                'sitting': '🪑',
                'standing': '🧍',
                'lying': '🛏️',
                'sleeping': '😴',
                'unknown': '❓'
            };

            const stateNames = {
                'sitting': 'Sitting',
                'standing': 'Standing',
                'lying': 'Lying',
                'sleeping': 'Sleeping',
                'unknown': 'Unknown'
            };

            let matchIcon = suggestion.match ? '✅' : '💡';
            let matchClass = suggestion.match ? 'match' : 'mismatch';

            content.innerHTML = `
                <div class="behavior-prediction">
                    <div class="prediction-row">
                        <div class="prediction-item-behavior">
                            <h4>📊 Historical Pattern</h4>
                            <p class="prediction-value-behavior">
                                ${stateIcons[prediction.predicted_state]} ${stateNames[prediction.predicted_state]}
                            </p>
                            <p class="prediction-confidence">Confidence: ${(prediction.confidence * 100).toFixed(0)}%</p>
                            <p class="prediction-explanation">${prediction.explanation}</p>
                        </div>
                        <div class="prediction-item-behavior">
                            <h4>🎯 Current Actual State</h4>
                            <p class="prediction-value-behavior">
                                ${stateIcons[currentState]} ${stateNames[currentState]}
                            </p>
                            <p class="prediction-match ${matchClass}">
                                ${matchIcon} ${suggestion.match ? 'Matches Pattern' : 'Differs from Pattern'}
                            </p>
                        </div>
                    </div>
                    <div class="smart-suggestion ${suggestion.priority}">
                        <p>${suggestion.message}</p>
                    </div>
                </div>
            `;
        } else {
            content.innerHTML = `<p>📊 Behavior predictions will be available after collecting more data...</p>`;
        }
    } catch (error) {
        console.error('Failed to fetch behavior prediction:', error);
        document.getElementById('behaviorPredictionContent').innerHTML =
            `<p>Unable to load behavior prediction</p>`;
    }
}

// Update daily routine
async function updateDailyRoutine() {
    try {
        const response = await fetch('/api/behavior/daily_routine');
        const result = await response.json();

        const content = document.getElementById('routineContent');

        if (result.success && result.data.routine) {
            const routine = result.data;

            const stateNames = {
                'sitting': 'Sitting/Working',
                'standing': 'Standing/Active',
                'lying': 'Lying/Resting',
                'sleeping': 'Sleeping',
                'unknown': 'Unknown'
            };

            let routineHTML = '<div class="routine-timeline">';
            routine.routine.forEach(item => {
                const stateName = stateNames[item.typical_state] || item.typical_state;
                const confidence = (item.confidence * 100).toFixed(0);

                routineHTML += `
                    <div class="routine-item">
                        <div class="routine-time">${item.time_range}</div>
                        <div class="routine-state">${stateName}</div>
                        <div class="routine-bar" style="width: ${confidence}%"></div>
                    </div>
                `;
            });
            routineHTML += '</div>';
            routineHTML += `<p class="routine-summary">${routine.summary}</p>`;
            routineHTML += `<p class="routine-note">Based on ${routine.analysis_period} of data</p>`;

            content.innerHTML = routineHTML;
        } else {
            content.innerHTML = `<p>Routine summary will be generated after collecting more data...</p>`;
        }
    } catch (error) {
        console.error('Failed to fetch daily routine:', error);
        document.getElementById('routineContent').innerHTML =
            `<p>Unable to load routine data</p>`;
    }
}

// Refresh all data
async function refreshAllData() {
    await Promise.all([
        updateCurrentSession(),
        updateTodayStats(),
        updateSittingDetails(),
        checkProlongedSitting(),
        updateWeeklyTrendChart(),
        updatePredictions(),
        updateAnomalyDetection(),
        updateBehaviorPrediction(),
        updateDailyRoutine()
    ]);

    updateLastUpdateTime();
    startRefreshCountdown();
}

// Initialize
async function init() {
    console.log('Initializing dashboard...');

    // Load data for the first time
    await refreshAllData();

    // Set up auto-refresh (every 30 seconds)
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }
    refreshInterval = setInterval(refreshAllData, 30000);

    console.log('Dashboard initialized');
}

// Initialize after page load
document.addEventListener('DOMContentLoaded', init);

// Clean up timers on page unload
window.addEventListener('beforeunload', () => {
    if (refreshInterval) clearInterval(refreshInterval);
    if (countdownInterval) clearInterval(countdownInterval);
});
