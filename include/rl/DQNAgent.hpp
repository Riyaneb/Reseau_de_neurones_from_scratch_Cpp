#ifndef RESEAU_DE_NEURONES_FROM_SCRATCH_C_DQNAGENT_HPP
#define RESEAU_DE_NEURONES_FROM_SCRATCH_C_DQNAGENT_HPP

#include "nn/Network.hpp"
#include "nn/Optimizers.hpp"
#include "rl/ReplayBuffer.hpp"
#include "rl/Transition.hpp"
#include "rl/EpsilonSchedule.hpp"
#include "env/Environment.hpp"
#include "nn/Losses.hpp"
#include "nn/DenseLayer.hpp"
#include "nn/Activations.hpp"

struct ConfigurationAgent {
    std::uint32_t seed = 0;
    Index amount_neural = 64;
    Scalar value_gamma = 0.95;
    Index value_size_batch = 64;
    Index value_min_amount_transition = 2000;
    Index value_update_intervale = 1000;
    Index buffer_size = 50000;
    Scalar learning_rate = 0.0005;
};


class DQNAgent {
public:
    DQNAgent(Index size_observation, Index am_action, ConfigurationAgent configuration);
    Index choice_action(Matrix const &observation);
    Index choice_action_test(Matrix const &observation);
    void add_buffer(Transition const &transition);
    void sync_target();
    void learn();
    void load_network(std::string const &nom, bool in_gpu=false);

    Scalar get_loss() const;
    Scalar actual_epsilon() const;
    Index get_size_buffer() const;
    Index get_step_count() const;
    Network get_network();
private:
    Random gen;
    Index amount_observation;
    Index amount_action;
    Network main_network;
    Network target_network;
    PairParameters parameters;
    std::unique_ptr<Optimizer> optimizer;
    QuadraLoss loss_fn;
    ReplayBuffer replay_buffer;
    EpsilonSchedule epsilon_schedule;
    Index step_count;
    Scalar gamma;
    Index size_batch;
    Index min_amount_transition;
    Index target_update_interval;
    Scalar loss_value;
};

#endif //RESEAU_DE_NEURONES_FROM_SCRATCH_C_DQNAGENT_HPP
