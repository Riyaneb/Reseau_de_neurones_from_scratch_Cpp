#ifndef RESEAUDENEURONEFROMSCRATCH_LOGGER_HPP
#define RESEAUDENEURONEFROMSCRATCH_LOGGER_HPP
#include "rl/DQNAgent.hpp"
#include "env/SnakeEnv.hpp"


void print_configuration(ConfigurationAgent const &config);
void clear();
void delay(int millisecondes);
void comparaison_matrix(Matrix const &result, Matrix const &expected, std::string const &nom);
Scalar eval_snake(DQNAgent &agent, SnakeEnv &env, Index nb_games);
void update_log(std::ofstream &log_file, std::deque<Scalar> &recent_scores, Scalar &sum_scores, Index episode, Scalar score, Scalar epsilon, Scalar loss);

#endif //RESEAUDENEURONEFROMSCRATCH_LOGGER_HPP
