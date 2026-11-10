#include "rl/DQNAgent.hpp"

#include "io/Serializer.hpp"

DQNAgent::DQNAgent(Index size_observation, Index am_action, ConfigurationAgent configuration)
: gen(configuration.seed), amount_observation(size_observation), amount_action(am_action), replay_buffer(configuration.buffer_size,configuration.seed), step_count(0), gamma(configuration.value_gamma), size_batch(configuration.value_size_batch), min_amount_transition(configuration.value_min_amount_transition), target_update_interval(configuration.value_update_intervale), loss_value(0)
{
    auto ptrDense1 = std::make_unique<DenseLayer>(size_observation, configuration.amount_neural, gen, 1);
    main_network.add(std::move(ptrDense1));
    main_network.add(std::make_unique<ReLU>(configuration.amount_neural));

    auto ptrDense2 = std::make_unique<DenseLayer>(configuration.amount_neural, am_action, gen);
    main_network.add(std::move(ptrDense2));
    main_network.add(std::make_unique<Identity>(am_action));

    parameters = main_network.get_parameters();
    optimizer = std::make_unique<Adam>(parameters , configuration.learning_rate);

    target_network = main_network.clone();
}

Index DQNAgent::choice_action(Matrix const &observation) {
    Scalar epsilon = epsilon_schedule.exponential_decay(step_count);
    Scalar rand = gen.randomizeUniform(0,1);
    if (rand < epsilon) {
        return gen.randomizeInt(0,amount_action-1);
    }
    else {
        Matrix result = main_network.forward(observation);
        return result.max_index_value_row(0);
    }
}

Index DQNAgent::choice_action_test(Matrix const &observation) {
    Matrix result = main_network.forward(observation);
    return result.max_index_value_row(0);
}

void DQNAgent::add_buffer(Transition const &transition) {
    replay_buffer.add_transition(transition);
}

void DQNAgent::sync_target() {
    target_network = main_network.clone();
}

Scalar DQNAgent::get_loss() const {
    return loss_value;
}

Scalar DQNAgent::actual_epsilon() const{
    return epsilon_schedule.exponential_decay(step_count);
}

Index DQNAgent::get_size_buffer() const{
    return replay_buffer.get_size();
}

Index DQNAgent::get_step_count() const{
    return step_count;
}

Network DQNAgent::get_network() {
    return main_network.clone();
}

void DQNAgent::load_network(std::string const &nom, bool in_gpu) {
    Network net;
    load(net, nom, in_gpu);
    main_network = std::move(net);
    sync_target();
}

void DQNAgent::learn() {
    if (!replay_buffer.is_enough(min_amount_transition)){
        return;
    }
    step_count++;
    auto batch = replay_buffer.get_transitions(size_batch);
    Matrix start_state(size_batch,amount_observation);
    Matrix end_state(size_batch,amount_observation);
    for (Index i = 0; i < size_batch; i++) {
        for (Index j = 0; j < amount_observation; j++) {
            start_state.get_value(i,j) = batch[i].start_state.get_value(0,j);
            end_state.get_value(i,j) = batch[i].end_state.get_value(0,j);
        }
    }
    Matrix q_values = main_network.forward(start_state);
    Matrix next_q_values = target_network.forward(end_state);
    Matrix target = q_values;
    for (Index i = 0; i < size_batch; i++) {
        Index action = batch[i].action;
        bool end = batch[i].end_game;
        Scalar rew = batch[i].reward;
        if (end) {
            target.get_value(i,action) = rew;
        }
        else {
            target.get_value(i,action) = rew + gamma * next_q_values.max_value_row(i);
        }
    }
    main_network.set_gradients_zero();
    loss_value = loss_fn.calculate_loss(q_values, target);
    main_network.backward(loss_fn.calculate_gradients_loss(q_values, target));
    optimizer->update_parameters(parameters);
    if (step_count % target_update_interval == 0) {
        sync_target();
    }
}
