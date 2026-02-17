#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "raylib.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <time.h>

#define USE_GAMMA

#define EMPTY 0
#define FOOD 1
#define CORPSE 2
#define WALL 3

// Multi-objective: 3 reward components
#define REWARD_DIM 3
#define REWARD_FOOD_IDX 0
#define REWARD_CORPSE_IDX 1
#define REWARD_DEATH_IDX 2

// Uniform Dirichlet for equal preference initialization
const double dirichlet_alpha[] = {1.0, 1.0, 1.0};

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float scalarized_episode_return;
    float discounted_episode_return;
    float discounted_scalarized_episode_return;
    float episode_return_food;
    float episode_return_corpse;
    float episode_return_death;
    float discounted_episode_return_food;
    float discounted_episode_return_corpse;
    float discounted_episode_return_death;
    float weight_food;
    float weight_corpse;
    float weight_death;
    float episode_length;
    float n;
};

typedef struct Client Client;
typedef struct CSnakeMO CSnakeMO;
struct CSnakeMO {
    char* observations;
    int* actions;
    float* rewards;
    float* weights;
    unsigned char* terminals;
    Log log;
    Log* snake_logs;
    char* grid;
    int* snake;
    int* snake_lengths;
    int* snake_ptr;
    int* snake_lifetimes;
    int* snake_colors;
    int num_snakes;
    int width;
    int height;
    int max_snake_length;
    int food;
    int vision;
    int window;
    int obs_size;
    unsigned char leave_corpse_on_death;
    float reward_food;
    float reward_corpse;
    float reward_death;
    float weight_food;
    float weight_corpse;
    float weight_death;
    int tick;
    int max_ticks;
    int max_ticks_offset_mod;
    int current_max_ticks;
    bool freeze_on_done;
    bool done;
    int cell_size;
    double gamma;
    double gamma_t;
    bool manual_weights;
    gsl_rng* gsl_rng;
    Client* client;
};

/**
 * Add a snake's log to the main log when the snake's episode ends (dies or hits a wall).
 * This should only be called during termination/truncation conditions for a specific snake.
 * Accumulates the snake's stats into the main log and resets the snake's individual log.
 */
void add_log(CSnakeMO* env, int snake_id) {
    env->log.perf += env->snake_logs[snake_id].perf;
    env->log.score += env->snake_logs[snake_id].score;
    env->log.episode_return += env->snake_logs[snake_id].episode_return;
    env->log.scalarized_episode_return += env->snake_logs[snake_id].scalarized_episode_return;
    env->log.discounted_episode_return += env->snake_logs[snake_id].discounted_episode_return;
    env->log.discounted_scalarized_episode_return += env->snake_logs[snake_id].discounted_scalarized_episode_return;
    env->log.discounted_episode_return_food += env->snake_logs[snake_id].discounted_episode_return_food;
    env->log.discounted_episode_return_corpse += env->snake_logs[snake_id].discounted_episode_return_corpse;
    env->log.discounted_episode_return_death += env->snake_logs[snake_id].discounted_episode_return_death;
    env->log.episode_return_food += env->snake_logs[snake_id].episode_return_food;
    env->log.episode_return_corpse += env->snake_logs[snake_id].episode_return_corpse;
    env->log.episode_return_death += env->snake_logs[snake_id].episode_return_death;
    env->log.weight_food += env->snake_logs[snake_id].weight_food;
    env->log.weight_corpse += env->snake_logs[snake_id].weight_corpse;
    env->log.weight_death += env->snake_logs[snake_id].weight_death;
    env->log.episode_length += env->snake_logs[snake_id].episode_length;
    env->log.n += 1;
}

void init_csnake(CSnakeMO* env) {
    env->grid = (char*)calloc(env->width*env->height, sizeof(char));
    env->snake = (int*)calloc(env->num_snakes*2*env->max_snake_length, sizeof(int));
    env->snake_lengths = (int*)calloc(env->num_snakes, sizeof(int));
    env->snake_ptr = (int*)calloc(env->num_snakes, sizeof(int));
    env->snake_lifetimes = (int*)calloc(env->num_snakes, sizeof(int));
    env->snake_colors = (int*)calloc(env->num_snakes, sizeof(int));
    env->snake_logs = (Log*)calloc(env->num_snakes, sizeof(Log));
    env->tick = 0;
    env->gamma_t = env->gamma;
    env->manual_weights = false;
    env->client = NULL;
    
    // Initialize GSL RNG if not already initialized
    if (env->gsl_rng == NULL) {
        env->gsl_rng = gsl_rng_alloc(gsl_rng_default);
        gsl_rng_set(env->gsl_rng, time(NULL));
    }
    
    env->snake_colors[0] = 7;
    for (int i = 1; i<env->num_snakes; i++)
        env->snake_colors[i] = i%4 + 4; // Randomize snake colors
}

void c_close(CSnakeMO* env) {
    free(env->grid);
    free(env->snake);
    free(env->snake_lengths);
    free(env->snake_ptr);
    free(env->snake_lifetimes);
    free(env->snake_colors);
    free(env->snake_logs);
    if (env->gsl_rng) {
        gsl_rng_free(env->gsl_rng);
    }
}

void allocate_csnake(CSnakeMO* env) {
    int obs_size = (2*env->vision + 1) * (2*env->vision + 1);
    env->observations = (char*)calloc(env->num_snakes*obs_size, sizeof(char));
    env->actions = (int*)calloc(env->num_snakes, sizeof(int));
    env->rewards = (float*)calloc(env->num_snakes * REWARD_DIM, sizeof(float));
    env->weights = (float*)calloc(env->num_snakes * REWARD_DIM, sizeof(float));
    env->terminals = (unsigned char*)calloc(env->num_snakes, sizeof(unsigned char));
    env->done = false;
    init_csnake(env);
}

void free_csnake(CSnakeMO* env) {
    c_close(env);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->weights);
    free(env->terminals);
}

void compute_observations(CSnakeMO* env) {
    for (int i = 0; i < env->num_snakes; i++) {
        int head_ptr = i*2*env->max_snake_length + 2*env->snake_ptr[i];
        int r_offset = env->snake[head_ptr] - env->vision;
        int c_offset = env->snake[head_ptr+1] - env->vision;
        for (int r = 0; r < 2 * env->vision + 1; r++) {
            for (int c = 0; c < 2 * env->vision + 1; c++) {
                env->observations[i*env->obs_size + r*env->window + c] = env->grid[
                    (r_offset + r)*env->width + c_offset + c];
            }
        }
    }
}

void delete_snake(CSnakeMO* env, int snake_id) {
    while (env->snake_lengths[snake_id] > 0) {
        int head_ptr = env->snake_ptr[snake_id];
        int head_offset = 2*env->max_snake_length*snake_id + 2*head_ptr;
        int head_r = env->snake[head_offset];
        int head_c = env->snake[head_offset + 1];
        if (env->leave_corpse_on_death && env->snake_lengths[snake_id] % 2 == 0)
            env->grid[head_r*env->width + head_c] = CORPSE;
        else
            env->grid[head_r*env->width + head_c] = EMPTY;

        env->snake[head_offset] = -1;
        env->snake[head_offset + 1] = -1;
        env->snake_lengths[snake_id]--;
        if (head_ptr == 0)
            env->snake_ptr[snake_id] = env->max_snake_length - 1;
        else
            env->snake_ptr[snake_id]--;
    }
}

void spawn_snake(CSnakeMO* env, int snake_id) {
    int head_r, head_c, tile, grid_idx;
    delete_snake(env, snake_id);
    do {
        head_r = rand() % (env->height - 1);
        head_c = rand() % (env->width - 1);
        grid_idx = head_r*env->width + head_c;
        tile = env->grid[grid_idx];
    } while (tile != EMPTY && tile != CORPSE);
    int snake_offset = 2*env->max_snake_length*snake_id;
    env->snake[snake_offset] = head_r;
    env->snake[snake_offset + 1] = head_c;
    env->snake_lengths[snake_id] = 1;
    env->snake_ptr[snake_id] = 0;
    env->snake_lifetimes[snake_id] = 0;
    env->grid[grid_idx] = env->snake_colors[snake_id];
    env->snake_logs[snake_id] = (Log){0};
}

void spawn_food(CSnakeMO* env) {
    int idx, tile;
    do {
        int r = rand() % (env->height - 1);
        int c = rand() % (env->width - 1);
        idx = r*env->width + c;
        tile = env->grid[idx];
    } while (tile != EMPTY && tile != CORPSE);
    env->grid[idx] = FOOD;
}

void c_reset(CSnakeMO* env) {
    if (env->freeze_on_done && env->done) {
        return;
    }
    
    env->window = 2*env->vision+1;
    env->obs_size = env->window*env->window;
    env->tick = 0;
    env->current_max_ticks = env->max_ticks + (rand() % env->max_ticks_offset_mod) - env->max_ticks_offset_mod / 2;
    env->gamma_t = env->gamma;
    env->log = (Log){0};

    // Clear the full board (including corpses/snakes) before rebuilding walls.
    memset(env->grid, EMPTY, env->width*env->height*sizeof(char));
    
    // Prevent spawn_snake()->delete_snake() from writing CORPSE tiles from the previous episode.
    // We are doing a full reset, so we want a clean board.
    for (int i = 0; i < env->num_snakes; i++) {
        env->snake_lengths[i] = 0;
        env->snake_ptr[i] = 0;
        env->snake_lifetimes[i] = 0;
    }

    for (int i = 0; i < env->num_snakes; i++)
        env->snake_logs[i] = (Log){0};

    for (int r = 0; r < env->vision; r++) {
        for (int c = 0; c < env->width; c++)
            env->grid[r*env->width + c] = WALL;
    }
    for (int r = env->height - env->vision; r < env->height; r++) {
        for (int c = 0; c < env->width; c++)
            env->grid[r*env->width + c] = WALL;
    }
    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->vision; c++)
            env->grid[r*env->width + c] = WALL;
        for (int c = env->width - env->vision; c < env->width; c++)
            env->grid[r*env->width + c] = WALL;
    }
    for (int i = 0; i < env->num_snakes; i++)
        spawn_snake(env, i);
    for (int i = 0; i < env->food; i++)
        spawn_food(env);

    compute_observations(env);

    // Sample new reward weights for each episode (unless manually set)
    if (env->manual_weights) {
        return;
    }
        
    double shared_weight_buffer[REWARD_DIM];
    gsl_ran_dirichlet(env->gsl_rng, REWARD_DIM, dirichlet_alpha, shared_weight_buffer);
    for (int j = 0; j < env->num_snakes; j++) {
        for (int i = 0; i < REWARD_DIM; i++) {
            env->weights[j * REWARD_DIM + i] = (float)shared_weight_buffer[i];
        }
    }
}

void step_snake(CSnakeMO* env, int i) {
    env->snake_logs[i].episode_length += 1;
    int atn = env->actions[i];
    int dr = 0;
    int dc = 0;
    switch (atn) {
        case 0: dr = -1; break; // up
        case 1: dr = 1; break;  // down
        case 2: dc = -1; break; // left
        case 3: dc = 1; break;  // right
    }

    int head_ptr = env->snake_ptr[i];
    int snake_offset = 2*env->max_snake_length*i;
    int head_offset = snake_offset + 2*head_ptr;
    int next_r = dr + env->snake[head_offset];
    int next_c = dc + env->snake[head_offset + 1];

    // Disallow moving into own neck
    int prev_head_offset = head_offset - 2;
    if (prev_head_offset < 0)
        prev_head_offset += 2*env->max_snake_length;
    int prev_r = env->snake[prev_head_offset];
    int prev_c = env->snake[prev_head_offset + 1];
    if (prev_r == next_r && prev_c == next_c) {
        next_r = env->snake[head_offset] - dr;
        next_c = env->snake[head_offset + 1] - dc;
    }

    float weight_food = env->weights[i * REWARD_DIM + REWARD_FOOD_IDX];
    float weight_corpse = env->weights[i * REWARD_DIM + REWARD_CORPSE_IDX];
    float weight_death = env->weights[i * REWARD_DIM + REWARD_DEATH_IDX];

    env->snake_logs[i].weight_food = weight_food;
    env->snake_logs[i].weight_corpse = weight_corpse;
    env->snake_logs[i].weight_death = weight_death;

    float reward_food = 0.0f;
    float reward_corpse = 0.0f;
    float reward_death = 0.0f;
    
    // Initialize reward components to zero
    for (int j = 0; j < REWARD_DIM; j++) {
        env->rewards[i * REWARD_DIM + j] = 0.0f;
    }

    int tile = env->grid[next_r*env->width + next_c];
    if (tile >= WALL) {
        reward_death = env->reward_death;
        env->rewards[i * REWARD_DIM + REWARD_DEATH_IDX] = reward_death;
        env->snake_logs[i].episode_return_death += reward_death;
        float scalarized_reward = reward_death * weight_death;
        env->snake_logs[i].episode_return += reward_death;
        env->snake_logs[i].scalarized_episode_return += scalarized_reward;
        env->snake_logs[i].discounted_episode_return += env->gamma_t * reward_death;
        env->snake_logs[i].discounted_scalarized_episode_return += env->gamma_t * scalarized_reward;        
        env->snake_logs[i].discounted_episode_return_death += env->gamma_t * reward_death;
        env->snake_logs[i].score = env->snake_lengths[i];
        env->snake_logs[i].perf = env->snake_logs[i].score / env->snake_logs[i].episode_length;
        add_log(env, i);
        spawn_snake(env, i);
        return;
    }

    head_ptr++; // Circular buffer
    if (head_ptr >= env->max_snake_length)
        head_ptr = 0;
    head_offset = snake_offset + 2*head_ptr;
    env->snake[head_offset] = next_r;
    env->snake[head_offset + 1] = next_c;
    env->snake_ptr[i] = head_ptr;
    env->snake_lifetimes[i]++;

    bool grow;
    if (tile == FOOD) {
        reward_food = env->reward_food;
        spawn_food(env);
        grow = true;
    } else if (tile == CORPSE) {
        reward_corpse = env->reward_corpse;
        grow = true;
    } else {
        grow = false;
    }

    env->rewards[i * REWARD_DIM + REWARD_FOOD_IDX] = reward_food;
    env->rewards[i * REWARD_DIM + REWARD_CORPSE_IDX] = reward_corpse;
    env->snake_logs[i].episode_return += reward_food + reward_corpse;
    env->snake_logs[i].episode_return_food += reward_food;
    env->snake_logs[i].episode_return_corpse += reward_corpse;
    env->snake_logs[i].discounted_episode_return += env->gamma_t * (reward_food + reward_corpse);
    env->snake_logs[i].discounted_episode_return_food += env->gamma_t * reward_food;
    env->snake_logs[i].discounted_episode_return_corpse += env->gamma_t * reward_corpse;
    float scalarized_reward = reward_food * weight_food + reward_corpse * weight_corpse;
    env->snake_logs[i].scalarized_episode_return += scalarized_reward;
    env->snake_logs[i].discounted_scalarized_episode_return += env->gamma_t * scalarized_reward;
    
    int snake_length = env->snake_lengths[i];
    if (grow && snake_length < env->max_snake_length - 1) {
        env->snake_lengths[i]++;
    } else {
        int tail_ptr = head_ptr - snake_length;
        if (tail_ptr < 0) // Circular buffer
            tail_ptr = env->max_snake_length + tail_ptr;
        int tail_r = env->snake[snake_offset + 2*tail_ptr];
        int tail_c = env->snake[snake_offset + 2*tail_ptr + 1];
        int tail_offset = 2*env->max_snake_length*i + 2*tail_ptr;
        env->snake[tail_offset] = -1;
        env->snake[tail_offset + 1] = -1;
        env->grid[tail_r*env->width + tail_c] = EMPTY;
    }
    env->grid[next_r*env->width + next_c] = env->snake_colors[i];
}

void c_step(CSnakeMO* env){
    if (env->freeze_on_done && env->done) {
        return;
    }
    
    env->tick++;
    env->gamma_t *= env->gamma;

    // Clear terminals each step. We only raise terminals on the MAX_TICKS boundary.
    memset(env->terminals, 0, env->num_snakes);

    for (int i = 0; i < env->num_snakes; i++)
        step_snake(env, i);

    // Fixed-horizon truncation: end all ongoing snake episodes and reset the board.
    if (env->current_max_ticks > 0 && env->tick >= env->current_max_ticks) {
        for (int i = 0; i < env->num_snakes; i++) {
            // Avoid double-counting snakes that already died and respawned this step.
            if (env->snake_logs[i].episode_length > 0) {
                env->snake_logs[i].score = env->snake_lengths[i];
                env->snake_logs[i].perf = env->snake_logs[i].score / env->snake_logs[i].episode_length;
                add_log(env, i);
            }
        }

        memset(env->terminals, 1, env->num_snakes);
        env->done = true;

        c_reset(env);
    }

    compute_observations(env);
}

// Raylib client
Color COLORS[] = {
    (Color){6, 24, 24, 255},
    (Color){0, 0, 255, 255},
    (Color){0, 128, 255, 255},
    (Color){128, 128, 128, 255},
    (Color){255, 0, 0, 255},
    (Color){255, 255, 255, 255},
    (Color){255, 85, 85, 255},
    (Color){170, 170, 170, 255},
    (Color){0, 255, 255, 255},
    (Color){255, 255, 0, 255},
};

typedef struct Client Client;
struct Client {
    int cell_size;
    int width;
    int height;
};

Client* make_client(int cell_size, int width, int height) {
    Client* client= (Client*)malloc(sizeof(Client));
    client->cell_size = cell_size;
    client->width = width;
    client->height = height;
    InitWindow(width*cell_size, height*cell_size, "PufferLib Snake MO");
    SetTargetFPS(10);
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void c_render(CSnakeMO* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    
    if (env->client == NULL) {
        env->client = make_client(env->cell_size, env->width, env->height);
    }
    
    Client* client = env->client;
    
    BeginDrawing();
    ClearBackground(COLORS[0]);
    int sz = client->cell_size;
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++){
            int tile = env->grid[y*env->width + x];
            if (tile != EMPTY)
                DrawRectangle(x*sz, y*sz, sz, sz, COLORS[tile]);
        }
    }
    EndDrawing();
}
