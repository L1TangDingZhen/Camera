// 全局变量
let todayPieChart = null;
let weeklyTrendChart = null;
let refreshCountdown = 30;
let refreshInterval = null;
let countdownInterval = null;

// 工具函数：格式化时长
function formatDuration(seconds) {
    if (seconds < 60) {
        return `${Math.floor(seconds)}秒`;
    } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        return `${minutes}分钟`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        if (minutes > 0) {
            return `${hours}小时${minutes}分钟`;
        } else {
            return `${hours}小时`;
        }
    }
}

// 工具函数：格式化小时数
function formatHours(seconds) {
    const hours = seconds / 3600;
    return `${hours.toFixed(1)}小时`;
}

// 更新当前会话信息
async function updateCurrentSession() {
    try {
        const response = await fetch('/api/stats/current');
        const result = await response.json();

        const content = document.getElementById('currentSessionContent');

        if (result.success && result.has_session) {
            const session = result.data;
            content.innerHTML = `
                <div class="session-info">
                    <div class="session-item">
                        <h4>当前状态</h4>
                        <p>${translateState(session.state)}</p>
                    </div>
                    <div class="session-item">
                        <h4>持续时长</h4>
                        <p>${formatDuration(session.duration)}</p>
                    </div>
                    ${session.zone ? `
                    <div class="session-item">
                        <h4>所在区域</h4>
                        <p>${session.zone}</p>
                    </div>
                    ` : ''}
                </div>
            `;
        } else {
            content.innerHTML = `
                <div class="session-info">
                    <div class="session-item">
                        <h4>当前状态</h4>
                        <p>无活动检测</p>
                    </div>
                </div>
            `;
        }
    } catch (error) {
        console.error('获取当前会话失败:', error);
    }
}

// 翻译状态
function translateState(state) {
    const stateMap = {
        'sitting': '🪑 坐姿',
        'standing': '🧍 站立',
        'lying': '🛏️ 躺卧',
        'sleeping': '😴 睡眠',
        'unknown': '❓ 未知'
    };
    return stateMap[state] || state;
}

// 更新今日统计
async function updateTodayStats() {
    try {
        const response = await fetch('/api/stats/today');
        const result = await response.json();

        if (result.success) {
            const stats = result.data;

            document.getElementById('todaySitting').textContent = formatHours(stats.sitting_duration);
            document.getElementById('todayStanding').textContent = formatHours(stats.standing_duration);
            document.getElementById('todayLying').textContent = formatHours(stats.lying_duration);
            document.getElementById('todaySessions').textContent = stats.total_sessions;

            // 更新饼图
            updateTodayPieChart(stats);
        }
    } catch (error) {
        console.error('获取今日统计失败:', error);
    }
}

// 更新坐姿详细统计
async function updateSittingDetails() {
    try {
        const response = await fetch('/api/stats/sitting');
        const result = await response.json();

        const content = document.getElementById('sittingDetailsContent');

        if (result.success) {
            const stats = result.data;

            content.innerHTML = `
                <div class="sitting-details-grid">
                    <div class="detail-item">
                        <h4>总时长</h4>
                        <p>${stats.total_duration_hours.toFixed(1)}小时</p>
                    </div>
                    <div class="detail-item">
                        <h4>会话次数</h4>
                        <p>${stats.session_count}次</p>
                    </div>
                    <div class="detail-item">
                        <h4>平均时长</h4>
                        <p>${formatDuration(stats.average_session_duration)}</p>
                    </div>
                    <div class="detail-item">
                        <h4>最长一次</h4>
                        <p>${formatDuration(stats.longest_session)}</p>
                    </div>
                </div>
            `;
        }
    } catch (error) {
        console.error('获取坐姿统计失败:', error);
    }
}

// 检查久坐警告
async function checkProlongedSitting() {
    try {
        const response = await fetch('/api/alert/prolonged_sitting?threshold=30');
        const result = await response.json();

        const alert = document.getElementById('prolongedSittingAlert');
        const message = document.getElementById('prolongedSittingMessage');

        if (result.success && result.data.alert) {
            const duration = Math.floor(result.data.current_duration_minutes);
            message.textContent = `您已持续坐姿${duration}分钟，建议起身活动！`;
            alert.style.display = 'flex';
        } else {
            alert.style.display = 'none';
        }
    } catch (error) {
        console.error('检查久坐警告失败:', error);
    }
}

// 创建/更新今日活动饼图
function updateTodayPieChart(stats) {
    const ctx = document.getElementById('todayPieChart');

    if (todayPieChart) {
        todayPieChart.destroy();
    }

    todayPieChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['坐姿', '站立', '躺卧'],
            datasets: [{
                label: '活动时长（小时）',
                data: [
                    stats.sitting_duration / 3600,
                    stats.standing_duration / 3600,
                    stats.lying_duration / 3600
                ],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(255, 206, 86, 0.8)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)'
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
                    text: '今日活动时长分布'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.label || '';
                            if (label) {
                                label += ': ';
                            }
                            label += context.parsed.toFixed(1) + '小时';
                            return label;
                        }
                    }
                }
            }
        }
    });
}

// 更新本周趋势图
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

        // 准备数据
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
                        label: '坐姿',
                        data: sittingData,
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        tension: 0.3,
                        fill: true
                    },
                    {
                        label: '站立',
                        data: standingData,
                        borderColor: 'rgb(54, 162, 235)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        tension: 0.3,
                        fill: true
                    },
                    {
                        label: '躺卧',
                        data: lyingData,
                        borderColor: 'rgb(255, 206, 86)',
                        backgroundColor: 'rgba(255, 206, 86, 0.2)',
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
                        text: '本周活动时长趋势（小时）'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                label += context.parsed.y + '小时';
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
                            text: '小时'
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('更新本周趋势图失败:', error);
    }
}

// 更新最后更新时间
function updateLastUpdateTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('zh-CN');
    document.getElementById('lastUpdate').textContent = timeString;
}

// 刷新倒计时
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

// 更新预测信息
async function updatePredictions() {
    try {
        // 获取下次坐姿预测
        const nextSittingResponse = await fetch('/api/prediction/next_sitting');
        const nextSittingResult = await nextSittingResponse.json();

        // 获取最佳提醒时间
        const reminderResponse = await fetch('/api/prediction/optimal_reminder');
        const reminderResult = await reminderResponse.json();

        const content = document.getElementById('predictionContent');

        if (nextSittingResult.success && reminderResult.success) {
            const prediction = nextSittingResult.data;
            const reminder = reminderResult.data;

            content.innerHTML = `
                <div class="prediction-grid">
                    <div class="prediction-item">
                        <h4>🎯 预测下次坐姿时长</h4>
                        <p class="prediction-value">${prediction.predicted_duration_minutes || 0}分钟</p>
                        <p class="prediction-confidence">置信度: ${(prediction.confidence * 100).toFixed(0)}%</p>
                        <p class="prediction-note">${prediction.recommendation}</p>
                    </div>
                    <div class="prediction-item">
                        <h4>⏰ 建议提醒间隔</h4>
                        <p class="prediction-value">${reminder.recommended_reminder_interval}分钟</p>
                        <p class="prediction-note">${reminder.pattern_description}</p>
                    </div>
                </div>
            `;
        } else {
            content.innerHTML = `<p>📊 累积更多数据后将提供预测...</p>`;
        }
    } catch (error) {
        console.error('获取预测失败:', error);
        document.getElementById('predictionContent').innerHTML =
            `<p>暂时无法获取预测数据</p>`;
    }
}

// 更新异常检测
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
                            <span class="label">今日坐姿:</span>
                            <span class="value">${anomaly.today_sitting_hours}小时</span>
                        </div>
                        <div class="anomaly-stat-item">
                            <span class="label">历史平均:</span>
                            <span class="value">${anomaly.average_sitting_hours}小时</span>
                        </div>
                        <div class="anomaly-stat-item">
                            <span class="label">偏差:</span>
                            <span class="value ${anomaly.deviation_percentage > 0 ? 'negative' : 'positive'}">
                                ${anomaly.deviation_percentage > 0 ? '+' : ''}${anomaly.deviation_percentage}%
                            </span>
                        </div>
                    </div>
                </div>
            `;
        } else {
            content.innerHTML = `<p>暂无足够数据进行异常检测</p>`;
        }
    } catch (error) {
        console.error('获取异常检测失败:', error);
    }
}

// 刷新所有数据
async function refreshAllData() {
    await Promise.all([
        updateCurrentSession(),
        updateTodayStats(),
        updateSittingDetails(),
        checkProlongedSitting(),
        updateWeeklyTrendChart(),
        updatePredictions(),
        updateAnomalyDetection()
    ]);

    updateLastUpdateTime();
    startRefreshCountdown();
}

// 初始化
async function init() {
    console.log('初始化仪表盘...');

    // 首次加载数据
    await refreshAllData();

    // 设置自动刷新（每30秒）
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }
    refreshInterval = setInterval(refreshAllData, 30000);

    console.log('仪表盘初始化完成');
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', init);

// 页面卸载时清理定时器
window.addEventListener('beforeunload', () => {
    if (refreshInterval) clearInterval(refreshInterval);
    if (countdownInterval) clearInterval(countdownInterval);
});
