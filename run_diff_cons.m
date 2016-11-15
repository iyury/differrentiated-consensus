%% построение графика 1
clear
disp('--- start simulation ---');
PLOT_GRAPHS_F_ON_GAMMA = true;
color = ['r', 'g', 'b', 'c', 'm', 'k'];
% число тактов
T = 10000;

adj_step = 100;

step_sizes = 0.01 : 0.01 : 0.4;
% число агентов
n = 20;
% по скольким x_0 идет усреднение
N = 10;
% помехи распределены равномерно в интервале [-0.1; 0.1]
a_w = -0.1;
b_w = 0.1;
w = a_w + (b_w - a_w) * rand(n, n, T);
% производительности - постоянные, равномерная случайная величина из [0.5, 1.5]
a_p = 0.5;
b_p = 1.5;
p = a_p + (b_p - a_p) * rand(1, n);

% генерация заданий
% количество приходящих заданий распределено по Пуассону
var_tz = n / 2;
% сложность приходящих заданий распределена равномерно
a_cz = 8;
b_cz = 12;
LOAD_TASKS = false;
if LOAD_TASKS == false
    incoming_tasks_num = poissrnd(var_tz, 1, T);%ceil(abs(var_tz * randn(1, T)));%
    tasks_num = sum(incoming_tasks_num);
    task_complexity = a_cz + (b_cz - a_cz) * rand(1, tasks_num);
    % tasks(1, :) - сложности заданий
    % tasks(2, :) - время поступления заданий
    tasks = zeros(2, tasks_num);
    j = 1;
    ti = 1;
    while ti <= T
        for l = 1 : incoming_tasks_num(ti)
            tasks(1, j) = task_complexity(j);
            tasks(2, j) = ti;
            j = j + 1;
        end
        ti = ti + 1;
    end
    save tasks.mat tasks
else
    load tasks.mat tasks
end
LOAD_TASKS = true;

init_tasks_num = ceil(rand(1, N) * 50 + 50);
initial_tasks = a_cz + (b_cz - a_cz) * rand(N, n, max(init_tasks_num));
save init_tasks.mat initial_tasks

% топология?
% A = [
%     0 0 0 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 0;
%     1 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0;
%     1 1 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0;
%     0 0 1 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0;
%     
%     0 0 1 0  0 0 0 1  0 0 0 0  0 0 0 0  0 0 0 0;
%     0 0 0 0  1 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0;
%     0 0 0 0  1 1 0 0  0 0 0 0  0 0 0 0  0 0 0 0;
%     0 0 0 0  0 0 1 0  0 0 0 0  0 0 0 0  0 0 0 0;
%     
%     0 0 0 0  0 0 1 0  0 0 0 1  0 0 0 0  0 0 0 0;
%     0 0 0 0  0 0 0 0  1 0 0 0  0 0 0 0  0 0 0 0;
%     0 0 0 0  0 0 0 0  1 1 0 0  0 0 0 0  0 0 0 0;
%     0 0 0 0  0 0 0 0  0 0 1 0  0 0 0 0  0 0 0 0;
%     
%     0 0 0 0  0 0 0 0  0 0 1 0  0 0 0 1  0 0 0 0;
%     0 0 0 0  0 0 0 0  0 0 0 0  1 0 0 0  0 0 0 0;
%     0 0 0 0  0 0 0 0  0 0 0 0  1 1 0 0  0 0 0 0;
%     0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 0  0 0 0 0;
%     
%     0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 0  0 0 0 1;
%     0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  1 0 0 0;
%     0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  1 1 0 0;
%     0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 0;
%     ];

% Topology1 = zeros(n, n);
% for i = 1 : n - 1
%     Topology1(i + 1, i) = 1;
%     Topology1(i, i + 1) = 1;
% end
% Topology1(1, n) = 1;
% Topology1(n, 1) = 1;
% A = Topology1;

% топология "двойное кольцо"
Topology1 = zeros(n, n);
for i = 1 : n
    Topology1(mod(i, n) + 1, i) = 1;
    Topology1(mod(i + 1, n) + 1, i) = 1;
end
Topology1 = Topology1 + Topology1';
A = Topology1;

% Topology1 = zeros(n, n);
% for i = 2 : n
%     Topology1(1, i) = 1;
%     Topology1(i, 1) = 1;
% end
% A = Topology1;
d_max = max(sum(A, 2));
%%
tic
if PLOT_GRAPHS_F_ON_GAMMA == true
%    step_sizes = 0.01 : 0.01 : 0.99;
    ss_size = size(step_sizes, 2);
    f_t = zeros(N, ss_size, T);
    % step size counter
    load tasks.mat tasks
    load init_tasks.mat initial_tasks
    for I = 1 : N
        SS = 1;
        for gamma = step_sizes
%             load tasks.mat tasks

            MAX_Q_LEN = T * var_tz;
            a_queues = zeros(n, MAX_Q_LEN);
            a_q_heads = ones(1, n);
            a_q_tails = ones(1, n);

%             load init_tasks.mat initial_tasks
            for i = 1 : n
                a_queues(i, a_q_tails(i) + (0 : (init_tasks_num(I) - 1))) = initial_tasks(I, i, 1 : init_tasks_num(I));
                a_q_tails(i) = a_q_tails(i) + init_tasks_num(I);
            end

            x_t = zeros(n, T);
            for i = 1 : n
                x_t(i, 1) = (a_q_tails(i) - a_q_heads(i)) / p(i);
            end
            % моделирование

            % tasks counter
            tc = 1;
            for t = 1 : adj_step
                % распределение заданий --- постановка приходящих заданий на агенты
                a_num = ceil(rand() * n);
                while tc <= size(tasks, 2) && tasks(2, tc) == t 

                    a_queues(a_num, a_q_tails(a_num)) = tasks(1, tc);
                    a_q_tails(a_num) = a_q_tails(a_num) + 1;
                    tc = tc + 1;
                end
                % выполненние заданий
                for i = 1 : n
                    % productitvity left
                    p_left = p(i);
                    while p_left > 0 && a_q_tails(i) - a_q_heads(i) > 0   %a_queues(i, a_q_heads(i)) > 0      
                        if p_left >= a_queues(i, a_q_heads(i))
                            p_left = p_left - a_queues(i, a_q_heads(i));
                            a_queues(i, a_q_heads(i)) = 0;
                            a_q_heads(i) = a_q_heads(i) + 1;
                        else
                            a_queues(i, a_q_heads(i)) = a_queues(i, a_q_heads(i)) - p_left;
                            p_left = 0;%break;
                        end            
                    end
                end
                % перераспределение заданий
                for i = 1 : n
                    x_t(i, t) = (a_q_tails(i) - a_q_heads(i)) / p(i);
                end
                % вычисление количества заданий для пересылки
                u = zeros(n, n);
                for i = 1 : n
                    for j = 1 : n
                        u(i, j) = gamma * A(i, j) * (x_t(j, t) + w(i, j, t) - x_t(i, t) - w(i, i, t));
                    end
                end
                u = round(u);
                % пересылка заданий
                for i = 1 : n
                    for j = 1 : n
                        if u(i, j) > 0 && a_q_tails(j) - a_q_heads(j) > 0
                            if a_q_tails(j) - a_q_heads(j) > u(i, j)
                                a_queues(i, a_q_tails(i) + (0 : (u(i, j) - 1))) = a_queues(j, a_q_tails(j) - (1 : (u(i, j))));
                                a_queues(j, a_q_tails(j) - (1 : (u(i, j)))) = 0;
                                a_q_tails(i) = a_q_tails(i) + u(i, j);
                                a_q_tails(j) = a_q_tails(j) - u(i, j);
                            else
                                a_queues(i, a_q_tails(i) + (0 : a_q_tails(j) - a_q_heads(j) - 1)) = a_queues(j, a_q_heads(j) : (a_q_tails(j) - 1));
                                a_queues(j, a_q_heads(j) : (a_q_tails(j) - 1)) = 0;
                                a_q_tails(i) = a_q_tails(i) + a_q_tails(j) - a_q_heads(j);
                                a_q_heads(j) = 1;
                                a_q_tails(j) = 1;
                            end
                        end
                    end
                end

                f_t(I, SS, t) = norm(x_t(:, t) - mean(x_t(:, t)));

            end
            SS = SS + 1;
        end
    end
end
disp('graph 1 data');
toc
% построение графиков
% hAgentStates = figure('Name', 'agent states');
% %axis([1 T 0 100])
% set(gca(hAgentStates), 'FontSize', 14);
% xlabel('T', 'FontSize', 20, 'FontAngle', 'italic');
% ylabel('x_t', 'FontSize', 20, 'FontAngle', 'italic', 'Rotation', 0);
% hold on
% for i = 1 : n
%     plot(1 : T, x_t(i, :), 'LineWidth', 2);
% end
% hold off

% hErrors = figure('Name', 'errors');
% %axis([1 T 0 100])
% set(gca(hAgentStates), 'FontSize', 14);
% xlabel('T', 'FontSize', 20, 'FontAngle', 'italic');
% ylabel('errors', 'FontSize', 20, 'FontAngle', 'italic', 'Rotation', 0);
% hold on
% plot(1 : T, error1, 'b', 'LineWidth', 2);
% plot(1 : T, error2, 'g', 'LineWidth', 2);
% plot(1 : T, error3, 'r', 'LineWidth', 2);
% hold off

% hf = figure('Name', 'f');
% %axis([1 T 0 100])
% set(gca(hf), 'FontSize', 14);
% xlabel('T', 'FontSize', 20, 'FontAngle', 'italic');
% ylabel('f_t', 'FontSize', 20, 'FontAngle', 'italic', 'Rotation', 0);
% hold on
% for SS = 1 : ss_size
%     plot(1 : T, f_t(SS, :), 'LineWidth', 2);
% end
% hold off

% gamma adjustment step
if PLOT_GRAPHS_F_ON_GAMMA == true
%     adj_step = 100;
    F = zeros(N, ss_size);%T / adj_step);
    for SS = 1 : ss_size
        for I = 1 : N
            F(I, SS) = 1 / adj_step * sum(f_t(I, SS, 1 : adj_step));
        end
    end
    F_n = sum(F, 1) / N;
    [F_n1_min, true_gamma_opt1_index] = min(F_n);
    true_gamma_opt1 = step_sizes(true_gamma_opt1_index);
end
% hF = figure('Name', 'F');
% %axis([1 T 0 100])
% set(gca(hF), 'FontSize', 14);
% xlabel('T', 'FontSize', 20, 'FontAngle', 'italic');
% ylabel('F', 'FontSize', 20, 'FontAngle', 'italic', 'Rotation', 0);
% hold on
% for SS = 1 : ss_size
%     plot(1 : T / adj_step, F(SS, :), 'LineWidth', 2);
% end
% hold off
%%
if PLOT_GRAPHS_F_ON_GAMMA == true
    hF1 = figure('Name', 'f1');
    set(gca(hF1), 'FontSize', 14);
    xlabel('\gamma', 'FontSize', 20, 'FontAngle', 'italic');
    y = ylabel('$\bar{F}$', 'Interpreter', 'LaTex', 'FontSize', 20, 'FontAngle', 'italic', 'Rotation', 0);
    set(y, 'Units', 'Normalized', 'Position', [-0.13, 0.5, 0]);
    hold on
    for I = 1 : N
        plot(step_sizes, F(I, :, 1), color(mod(I, length(color)) + 1), 'LineWidth', 2);
    end
    hold off
end

if PLOT_GRAPHS_F_ON_GAMMA == true
    hFn1 = figure('Name', 'F1');
    set(gca(hFn1), 'FontSize', 14);    
    xlabel('\gamma', 'FontSize', 20, 'FontAngle', 'italic');
    y = ylabel('$\cal{F}$', 'Interpreter', 'LaTex', 'FontSize', 20, 'FontAngle', 'italic', 'Rotation', 0);
    set(y, 'Units', 'Normalized', 'Position', [-0.13, 0.5, 0]);
    hold on
    plot(step_sizes, F_n, 'LineWidth', 2);
    hold off
end

% if PLOT_GRAPHS_F_ON_GAMMA == true
%     hf1 = figure('Name', 'f1');
%     set(gca(hf1), 'FontSize', 14);
%     xlabel('\gamma', 'FontSize', 20, 'FontAngle', 'italic');
%     ylabel('f_1', 'FontSize', 20, 'FontAngle', 'italic', 'Rotation', 0);
%     hold on
%     plot(step_sizes, f_t(:, adj_step), 'LineWidth', 2);
%     hold off
% end
%% построение графика 2
% число тактов
%T = 10000;
% число агентов
%n = 20;

% помехи распределены равномерно в интервале [-0.1; 0.1]
% a_w = -0.1;
% b_w = 0.1;
w = a_w + (b_w - a_w) * rand(n, n, T);
% производительности - постоянные, равномерная случайная величина из [0.5, 1.5]
%a_p = 0.5;
%b_p = 1.5;
%p = a_p + (b_p - a_p) * rand(1, n);

% генерация заданий
% количество приходящих заданий распределено по Пуассону
%var_tz = n / 2;
% сложность приходящих заданий распределена равномерно
%a_cz = 2;
%b_cz = 6;
LOAD_TASKS = false;
if LOAD_TASKS == false
    incoming_tasks_num = poissrnd(var_tz, 1, T);
    tasks_num = sum(incoming_tasks_num);
    task_complexity = a_cz + (b_cz - a_cz) * rand(1, tasks_num);
    % tasks(1, :) - сложности заданий
    % tasks(2, :) - время поступления заданий
    tasks = zeros(2, tasks_num);
    j = 1;
    ti = 1;
    while ti <= adj_step
        for l = 1 : incoming_tasks_num(ti)
            tasks(1, j) = task_complexity(j);
            tasks(2, j) = ti;
            j = j + 1;
        end
        ti = ti + 1;
    end
    save tasks2.mat tasks
else
    load tasks2.mat tasks
end
LOAD_TASKS = true;

% % топология?
% Topology2 = [
%     0 0 0 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 0;
%     1 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0;
%     1 1 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0;
%     0 0 1 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0;
%     
%     0 0 1 0  0 0 0 1  0 0 0 0  0 0 0 0  0 0 0 0;
%     0 0 0 0  1 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0;
%     0 0 0 0  1 1 0 0  0 0 0 0  0 0 0 0  0 0 0 0;
%     0 0 0 0  0 0 1 0  0 0 0 0  0 0 0 0  0 0 0 0;
%     
%     0 0 0 0  0 0 1 0  0 0 0 1  0 0 0 0  0 0 0 0;
%     0 0 0 0  0 0 0 0  1 0 0 0  0 0 0 0  0 0 0 0;
%     0 0 0 0  0 0 0 0  1 1 0 0  0 0 0 0  0 0 0 0;
%     0 0 0 0  0 0 0 0  0 0 1 0  0 0 0 0  0 0 0 0;
%     
%     0 0 0 0  0 0 0 0  0 0 1 0  0 0 0 1  0 0 0 0;
%     0 0 0 0  0 0 0 0  0 0 0 0  1 0 0 0  0 0 0 0;
%     0 0 0 0  0 0 0 0  0 0 0 0  1 1 0 0  0 0 0 0;
%     0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 0  0 0 0 0;
%     
%     0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 0  0 0 0 1;
%     0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  1 0 0 0;
%     0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  1 1 0 0;
%     0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 0;
%     ];
% A = Topology2;

% Topology2 = zeros(n, n);
% for i = 1 : n - 1
%     Topology2(i + 1, i) = 1;
% end
% Topology2(1, n) = 1;
% A = Topology2;

Topology2 = ones(n);
Topology2 = Topology2 - eye(n);
A = Topology2;

% A1 = zeros(n);
% for i = 2 : 2 : n - 2
%     A1(i + 2, i) = 1;
% end
% A1(2, n) = 1;
% A = A + A1;
d_max = max(sum(A, 2));
%%
tic
if PLOT_GRAPHS_F_ON_GAMMA == true
%    step_sizes = 0.01 : 0.01 : 0.99;
    ss_size = size(step_sizes, 2);
    f_t = zeros(N, ss_size, T);
    load tasks.mat tasks
    load init_tasks.mat initial_tasks
    % step size counter
    for I = 1 : N
        SS = 1;
        for gamma = step_sizes
%             load tasks.mat tasks

            MAX_Q_LEN = T * var_tz;
            a_queues = zeros(n, MAX_Q_LEN);
            a_q_heads = ones(1, n);
            a_q_tails = ones(1, n);

%             load init_tasks.mat initial_tasks
            for i = 1 : n
                a_queues(i, a_q_tails(i) + (0 : (init_tasks_num(I) - 1))) = initial_tasks(I, i, 1 : init_tasks_num(I));
                a_q_tails(i) = a_q_tails(i) + init_tasks_num(I);
            end

            x_t = zeros(n, T);

            for i = 1 : n
                x_t(i, 1) = (a_q_tails(i) - a_q_heads(i)) / p(i);
            end
            % моделирование

            % tasks counter
            tc = 1;
            for t = 1 : adj_step
                % распределение заданий --- постановка приходящих заданий на агенты
                a_num = ceil(rand() * n);
                while tc <= size(tasks, 2) && tasks(2, tc) == t 

                    a_queues(a_num, a_q_tails(a_num)) = tasks(1, tc);
                    a_q_tails(a_num) = a_q_tails(a_num) + 1;
                    tc = tc + 1;
                end
                % выполненние заданий
                for i = 1 : n
                    % productitvity left
                    p_left = p(i);
                    while p_left > 0 && a_q_tails(i) - a_q_heads(i) > 0   %a_queues(i, a_q_heads(i)) > 0      
                        if p_left >= a_queues(i, a_q_heads(i))
                            p_left = p_left - a_queues(i, a_q_heads(i));
                            a_queues(i, a_q_heads(i)) = 0;
                            a_q_heads(i) = a_q_heads(i) + 1;
                        else
                            a_queues(i, a_q_heads(i)) = a_queues(i, a_q_heads(i)) - p_left;
                            p_left = 0;%break;
                        end            
                    end
                end
                % перераспределение заданий
                for i = 1 : n
                    x_t(i, t) = (a_q_tails(i) - a_q_heads(i)) / p(i);
                end
                % вычисление количества заданий для пересылки
                u = zeros(n, n);
                for i = 1 : n
                    for j = 1 : n
                        u(i, j) = gamma * A(i, j) * (x_t(j, t) + w(i, j, t) - x_t(i, t) - w(i, i, t));
                    end
                end
                u = round(u);
                % пересылка заданий
                for i = 1 : n
                    for j = 1 : n
                        if u(i, j) > 0 && a_q_tails(j) - a_q_heads(j) > 0
                            if a_q_tails(j) - a_q_heads(j) > u(i, j)
                                a_queues(i, a_q_tails(i) + (0 : (u(i, j) - 1))) = a_queues(j, a_q_tails(j) - (1 : (u(i, j))));
                                a_queues(j, a_q_tails(j) - (1 : (u(i, j)))) = 0;
                                a_q_tails(i) = a_q_tails(i) + u(i, j);
                                a_q_tails(j) = a_q_tails(j) - u(i, j);
                            else
                                a_queues(i, a_q_tails(i) + (0 : a_q_tails(j) - a_q_heads(j) - 1)) = a_queues(j, a_q_heads(j) : (a_q_tails(j) - 1));
                                a_queues(j, a_q_heads(j) : (a_q_tails(j) - 1)) = 0;
                                a_q_tails(i) = a_q_tails(i) + a_q_tails(j) - a_q_heads(j);
                                a_q_heads(j) = 1;
                                a_q_tails(j) = 1;
                            end
                        end
                    end
                end

                f_t(I, SS, t) = norm(x_t(:, t) - mean(x_t(:, t)));

            end
            SS = SS + 1;
        end
    end
end
disp('graph 2 data');
toc
% построение графиков
% hAgentStates = figure('Name', 'agent states');
% %axis([1 T 0 100])
% set(gca(hAgentStates), 'FontSize', 14);
% xlabel('T', 'FontSize', 20, 'FontAngle', 'italic');
% ylabel('x_t', 'FontSize', 20, 'FontAngle', 'italic', 'Rotation', 0);
% hold on
% for i = 1 : n
%     plot(1 : T, x_t(i, :), 'LineWidth', 2);
% end
% hold off

% hErrors = figure('Name', 'errors');
% %axis([1 T 0 100])
% set(gca(hAgentStates), 'FontSize', 14);
% xlabel('T', 'FontSize', 20, 'FontAngle', 'italic');
% ylabel('errors', 'FontSize', 20, 'FontAngle', 'italic', 'Rotation', 0);
% hold on
% plot(1 : T, error1, 'b', 'LineWidth', 2);
% plot(1 : T, error2, 'g', 'LineWidth', 2);
% plot(1 : T, error3, 'r', 'LineWidth', 2);
% hold off

% hf = figure('Name', 'f');
% %axis([1 T 0 100])
% set(gca(hf), 'FontSize', 14);
% xlabel('T', 'FontSize', 20, 'FontAngle', 'italic');
% ylabel('f_t', 'FontSize', 20, 'FontAngle', 'italic', 'Rotation', 0);
% hold on
% for SS = 1 : ss_size
%     plot(1 : T, f_t(SS, :), 'LineWidth', 2);
% end
% hold off
%%
% gamma adjustment step
if PLOT_GRAPHS_F_ON_GAMMA == true
%     adj_step = 100;
    F = zeros(N, ss_size);%T / adj_step);
    for SS = 1 : ss_size
        for I = 1 : N
            F(I, SS) = 1 / adj_step * sum(f_t(I, SS, 1 : adj_step));
        end
    end
    F_n = sum(F, 1) / N;
    [F_n2_min, true_gamma_opt2_index] = min(F_n);
    true_gamma_opt2 = step_sizes(true_gamma_opt2_index);
end

% hF = figure('Name', 'F');
% %axis([1 T 0 100])
% set(gca(hF), 'FontSize', 14);
% xlabel('T', 'FontSize', 20, 'FontAngle', 'italic');
% ylabel('F', 'FontSize', 20, 'FontAngle', 'italic', 'Rotation', 0);
% hold on
% for SS = 1 : ss_size
%     plot(1 : T / adj_step, F(SS, :), 'LineWidth', 2);
% end
% hold off

if PLOT_GRAPHS_F_ON_GAMMA == true
    hF1 = figure('Name', 'f1');
    set(gca(hF1), 'FontSize', 14);
    xlabel('\gamma', 'FontSize', 20, 'FontAngle', 'italic');
    ylabel('$\bar{F}$', 'Interpreter', 'LaTex', 'FontSize', 20, 'FontAngle', 'italic', 'Rotation', 0);
    hold on
    for I = 1 : N
        plot(step_sizes, F(I, :, 1), color(mod(I, length(color)) + 1), 'LineWidth', 2);
    end
    hold off
end

if PLOT_GRAPHS_F_ON_GAMMA == true
    hFn1 = figure('Name', 'F1');
    set(gca(hFn1), 'FontSize', 14);
    xlabel('\gamma', 'FontSize', 20, 'FontAngle', 'italic');
    y = ylabel('$\cal{F}$', 'Interpreter', 'LaTeX', 'FontSize', 20, 'FontAngle', 'italic', 'Rotation', 0);
    set(y, 'Units', 'Normalized', 'Position', [-0.13, 0.5, 0]);
    hold on
    plot(step_sizes, F_n, 'LineWidth', 2);
    hold off
end

% if PLOT_GRAPHS_F_ON_GAMMA == true
%     hf1 = figure('Name', 'f1');
%     set(gca(hf1), 'FontSize', 14);
%     xlabel('\gamma', 'FontSize', 20, 'FontAngle', 'italic');
%     ylabel('f_1', 'FontSize', 20, 'FontAngle', 'italic', 'Rotation', 0);
%     hold on
%     plot(step_sizes, f_t(:, adj_step), 'LineWidth', 2);
%     hold off
% end
%% вычисление gamma_star
var_p = (b_p - a_p) ^ 2 / 12;
exp_p = (b_p - a_p) / 2;
% S = n * (var_z + var_r);
% 
var_w = (b_w - a_w) ^ 2 / 12;
% H = 2 * var_w * (norm(A, 2)^2);
% 
% %tmp = sort(re(eig(Q)));
% %R = 1;% - abs(tmp(2));
% %Delta = Re(max(eig(Q))) / R;
% 
% gamma_star = sqrt(S / H);
% %gamma_star = -S/H * Delta + sqrt((S/H*Delta)^2 + S/H);

var_tz = var_tz / n;
exp_tz = var_tz;
var_z = (1/var_p * var_tz + var_tz * 1/exp_p^2 + exp_tz^2 * 1/var_p);

B = Topology1;

a = sum(sum(B .* B)) * var_w;
b = n * var_z;
L = diag(sum(B, 2)) - B;
c = 2 * max(abs(eig(L)));
d = max(eig(L' * L));

gamma_star1 = (-b*d + sqrt(b^2 * d^2 + a*b*c^2)) / (a*c);

S = n*var_z;
H = var_w*sum(sum(B .* B));
D = max(eig(L' * L)) / ( 2 * max(abs(eig(L))));
g_s1 = -S/H*D + sqrt(S^2/H^2*D^2 + S/H);

B = Topology2;

a = sum(sum(B .* B)) * var_w;
b = n * var_z;
L = diag(sum(B, 2)) - B;
c = 2 * max(abs(eig(L)));
d = max(eig(L' * L));

gamma_star2 = (-b*d + sqrt(b^2 * d^2 + a*b*c^2)) / (a*c);

S = n*var_z;
H = var_w*sum(sum(B .* B));
D = max(eig(L' * L)) / ( 2 * max(abs(eig(L))));
g_s2 = -S/H*D + sqrt(S^2/H^2*D^2 + S/H);
%% gamma star
A = Topology1;
H = 2 * var_w * (max(eig(A' * A)));
S = n * var_z;
%Q = 0;
%Delta = max(eig(Q)) / R;
%gamma_opt = -S/H*Delta + sqrt(S^2/ H^2 * Delta^2 + S / H);
gamma_opt = sqrt(S/H);

A = Topology1;
L = diag(sum(A, 2)) - A;
g_opt1 = max(eig(L)) / max(eig(L' * L));


A = Topology2;
L = diag(sum(A, 2)) - A;
g_opt2 = max(eig(L)) / max(eig(L' * L));
%g_opt1 = (max(eig(L)) + sqrt(max(eig(L))^2 - max(L' * L))) / max(eig(L' * L));
%g_opt2 = (max(eig(L)) - sqrt(max(eig(L))^2 - max(L' * L))) / max(eig(L' * L));
%% L \bar

%beta_max = 
%% запуск алгоритма с адаптацией шага

load tasks.mat tasks
tmp = tasks;
load tasks2.mat tasks
tasks(2, :) = tasks(2, :) + T;
tasks = [tmp tasks];
T = 2 * T;
% число тактов
%T = 10000;
% число агентов
%n = 20;

% помехи распределены равномерно в интервале [-0.1; 0.1]
%a_w = -0.1;
%b_w = 0.1;
w = a_w + (b_w - a_w) * rand(n, n, T);
% производительности - постоянные, равномерная случайная величина из [0.5, 1.5]
%a_p = 0.5;
%b_p = 1.5;
%p = a_p + (b_p - a_p) * rand(1, n);

% генерация заданий
% количество приходящих заданий распределено по Пуассону
%var_tz = n / 10;
% сложность приходящих заданий распределена равномерно
%a_cz = 8;
%b_cz = 12;
%LOAD_TASKS = false;
% if LOAD_TASKS == false
%     incoming_tasks_num = poissrnd(var_tz, 1, T);
%     tasks_num = sum(incoming_tasks_num);
%     task_complexity = a_cz + (b_cz - a_cz) * rand(1, tasks_num);
%     % tasks(1, :) - сложности заданий
%     % tasks(2, :) - время поступления заданий
%     tasks = zeros(2, tasks_num);
%     j = 1;
%     ti = 1;
%     while ti <= T
%         for l = 1 : incoming_tasks_num(ti)
%             tasks(1, j) = task_complexity(j);
%             tasks(2, j) = ti;
%             j = j + 1;
%         end
%         ti = ti + 1;
%     end
%     save tasks.mat tasks
% else
%     load tasks.mat tasks
% end
% LOAD_TASKS = true;

% топология?
% A = [
%     0 0 0 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 0;
%     1 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0;
%     1 1 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0;
%     0 0 1 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0;
%     
%     0 0 1 0  0 0 0 1  0 0 0 0  0 0 0 0  0 0 0 0;
%     0 0 0 0  1 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0;
%     0 0 0 0  1 1 0 0  0 0 0 0  0 0 0 0  0 0 0 0;
%     0 0 0 0  0 0 1 0  0 0 0 0  0 0 0 0  0 0 0 0;
%     
%     0 0 0 0  0 0 1 0  0 0 0 1  0 0 0 0  0 0 0 0;
%     0 0 0 0  0 0 0 0  1 0 0 0  0 0 0 0  0 0 0 0;
%     0 0 0 0  0 0 0 0  1 1 0 0  0 0 0 0  0 0 0 0;
%     0 0 0 0  0 0 0 0  0 0 1 0  0 0 0 0  0 0 0 0;
%     
%     0 0 0 0  0 0 0 0  0 0 1 0  0 0 0 1  0 0 0 0;
%     0 0 0 0  0 0 0 0  0 0 0 0  1 0 0 0  0 0 0 0;
%     0 0 0 0  0 0 0 0  0 0 0 0  1 1 0 0  0 0 0 0;
%     0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 0  0 0 0 0;
%     
%     0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 0  0 0 0 1;
%     0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  1 0 0 0;
%     0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  1 1 0 0;
%     0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 0;
%     ];
A = Topology1;

% A1 = zeros(n);
% for i = 2 : 2 : n - 2
%     A1(i + 2, i) = 1;
% end
% A1(2, n) = 1;
% A = A + A1;

d_max = max(sum(A, 2));

f_t = zeros(1, T);
% step size counter
    
MAX_Q_LEN = T * var_tz;
a_queues = zeros(n, MAX_Q_LEN);
a_q_heads = ones(1, n);
a_q_tails = ones(1, n);


% начальная загрузка
load init_tasks.mat initial_tasks
for i = 1 : n
    a_queues(i, a_q_tails(i) + (0 : (init_tasks_num(I) - 1))) = initial_tasks(I, i, 1 : init_tasks_num(I));
    a_q_tails(i) = a_q_tails(i) + init_tasks_num(I);
end
%save init_tasks.mat initial_tasks


x_t = zeros(n, T);
% моделирование

for i = 1 : n
    x_t(i, 1) = (a_q_tails(i) - a_q_heads(i)) / p(i);
end



%adj_step = 100;
F = zeros(1, T / adj_step);
gamma_hist = zeros(1, T / adj_step);

alpha = 0.000015;
beta = 0.005;
gamma = 0.4;

% tasks counter
tc = 1;
tic
for t = 1 : T
    if t == T / 2
        A = Topology2;
        d_max = max(sum(A, 2));
    end
    % распределение заданий --- постановка приходящих заданий на агенты
    a_num = ceil(rand() * n);
    while tc <= size(tasks, 2) && tasks(2, tc) == t 
        
        a_queues(a_num, a_q_tails(a_num)) = tasks(1, tc);
        a_q_tails(a_num) = a_q_tails(a_num) + 1;
        tc = tc + 1;
    end
    % выполненние заданий
    for i = 1 : n
        % productitvity left
        p_left = p(i);
        while p_left > 0 && a_q_tails(i) - a_q_heads(i) > 0   %a_queues(i, a_q_heads(i)) > 0      
            if p_left >= a_queues(i, a_q_heads(i))
                p_left = p_left - a_queues(i, a_q_heads(i));
                a_queues(i, a_q_heads(i)) = 0;
                a_q_heads(i) = a_q_heads(i) + 1;
            else
                a_queues(i, a_q_heads(i)) = a_queues(i, a_q_heads(i)) - p_left;
                p_left = 0;%break;
            end            
        end
    end
    % перераспределение заданий
    for i = 1 : n
        x_t(i, t) = (a_q_tails(i) - a_q_heads(i)) / p(i);
    end
    % вычисление количества заданий для пересылки
    u = zeros(n, n);
    for i = 1 : n
        for j = 1 : n
            u(i, j) = gamma * A(i, j) * (x_t(j, t) + w(i, j, t) - x_t(i, t) - w(i, i, t));
        end
    end
    u = round(u);
    % пересылка заданий
    for i = 1 : n
        for j = 1 : n
            if u(i, j) > 0 && a_q_tails(j) - a_q_heads(j) > 0
                if a_q_tails(j) - a_q_heads(j) > u(i, j)
                    a_queues(i, a_q_tails(i) + (0 : (u(i, j) - 1))) = a_queues(j, a_q_tails(j) - (1 : (u(i, j))));
                    a_queues(j, a_q_tails(j) - (1 : (u(i, j)))) = 0;
                    a_q_tails(i) = a_q_tails(i) + u(i, j);
                    a_q_tails(j) = a_q_tails(j) - u(i, j);
                else
                    a_queues(i, a_q_tails(i) + (0 : a_q_tails(j) - a_q_heads(j) - 1)) = a_queues(j, a_q_heads(j) : (a_q_tails(j) - 1));
                    a_queues(j, a_q_heads(j) : (a_q_tails(j) - 1)) = 0;
                    a_q_tails(i) = a_q_tails(i) + a_q_tails(j) - a_q_heads(j);
                    a_q_heads(j) = 1;
                    a_q_tails(j) = 1;
                end
            end
        end
    end

    f_t(t) = norm(x_t(:, t) - mean(x_t(:, t)));
    if mod(t, adj_step) == 0
        t1 = t / adj_step;
        if mod(t, 2 * adj_step) ~= 0
            F_0 = f_t(t);%1 / adj_step * sum(f_t((1 + (t1 - 1) * adj_step) : t1 * adj_step));
            F(t1) = F_0;
            Delta = sign(rand() - 0.5);
            gamma = gamma + beta * Delta;            
            if gamma < 0
                gamma = 0.001;
            end
            if gamma >= 1 / d_max
                gamma = 1 / d_max - 0.001;
            end
            gamma_hist(t1) = gamma;
        else
            F_plus = f_t(t);%1 / adj_step * sum(f_t((1 + (t1 - 1) * adj_step) : t1 * adj_step)); 
            F(t1) = F_plus;
            gamma = gamma - alpha / beta * Delta * (F_plus - F_0);            
            if gamma < 0
                gamma = 0.001;
            end
            if gamma >= 1 / d_max
                gamma = 1 / d_max - 0.001;
            end
            gamma_hist(t1) = gamma;
        end
    end    
end
disp('adaptation');
toc
hGammaAdj = figure('Name', ['gamma_{adj} alpha = ' num2str(alpha) ' beta = ' num2str(beta)]);
set(gca(hGammaAdj), 'FontSize', 14);
xlabel('T_{adj}', 'FontSize', 20, 'FontAngle', 'italic');
ylabel('\gamma', 'FontSize', 20, 'FontAngle', 'italic', 'Rotation', 0);
hold on
plot(1 : T / adj_step, gamma_star1 * ones(1, T / adj_step), 'r');
plot(1 : T / adj_step, true_gamma_opt1 * ones(1, T / adj_step));
plot(1 : T / adj_step, gamma_star2 * ones(1, T / adj_step), 'r');
plot(1 : T / adj_step, true_gamma_opt2 * ones(1, T / adj_step));
plot(1 : T / adj_step, gamma_hist, 'LineWidth', 2);
hold off

% hFadj = figure('Name', 'F_{adj}');
% set(gca(hFadj), 'FontSize', 14);
% xlabel('T_{adj}', 'FontSize', 20, 'FontAngle', 'italic');
% ylabel('F_{adj}', 'FontSize', 20, 'FontAngle', 'italic', 'Rotation', 0);
% hold on
% plot(1 : T / adj_step, F, 'LineWidth', 2);
% hold off
disp('--- end simulation ---');