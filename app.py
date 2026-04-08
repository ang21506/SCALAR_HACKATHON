import gradio as gr
from env import SmartIrrigationEnv
from schemas import Action
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_plot(env):
    fig, ax = plt.subplots(figsize=(8, 4))
    if env is None or not hasattr(env, 'history'):
        return fig
        
    ax.axhspan(0.3, 0.8, color='green', alpha=0.2, label='Healthy Range')
    
    for i in range(env.num_plots):
        if i in env.history:
            ax.plot(env.history[i], label=f'Plot {i+1}')
            
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Steps (6 hours each)")
    ax.set_ylabel("Soil Moisture")
    ax.set_title("Soil Moisture over Time")
    ax.legend()
    plt.tight_layout()
    return fig

def create_env(task):
    env = SmartIrrigationEnv(task=task)
    obs = env.reset()
    return env, obs.model_dump()

def reset_ui(task):
    env, obs = create_env(task)
    return (
        env,
        gen_obs_text(obs, env),
        0.0,
        "{}",
        0, # step count
        "Episode Started - Awaiting Action",
        generate_plot(env)
    )

def gen_obs_text(obs, env):
    lines = [
        f"**Remaining Daily Quota:** {obs['daily_water_quota']:.1f}",
        f"**Pump Capacity:** {obs['pump_capacity']:.1f}",
        f"**Tariff Band:** {obs['tariff_band']}",
        f"**Rain Probability:** {obs['rain_probability']:.1%}",
        f"**Expected Rainfall:** {obs['expected_rainfall']:.1f} units",
        "---",
        "**Plots:**"
    ]
    for i in range(env.num_plots):
        lines.append(f"  - Plot {i+1} ({obs['crop_types'][i]}): Moisture = {obs['soil_moistures'][i]:.2f}, Growth Stage = {obs['growth_stages'][i]:.2f}")
    if env.num_plots < 5:
        lines.append(f"\n*(Note: Plots {env.num_plots+1}-5 are inactive in {env.task})*")
    return "\n".join(lines)

def step_ui(env, total_reward, step_count, *irrigation_amounts):
    if env is None:
        return env, "Please reset the environment first.", total_reward, "{}", step_count, "Error: Not initialized", None
        
    if step_count >= env.episode_length:
        return env, "Episode finished. Please reset.", total_reward, "{}", step_count, "Finished. Reset to continue.", generate_plot(env)

    # Truncate irrigation amounts to actual number of plots depending on task
    amounts = [int(a) for a in irrigation_amounts[:env.num_plots]]
    action = Action(irrigation_amounts=amounts)
    
    obs, reward, done, info = env.step(action)
    
    total_reward += reward.value
    step_count += 1
    
    status = f"Step {step_count}/{env.episode_length} completed."
    if done:
        status += " Episode Finished!"
        
    reward_dict = {
        "value": round(reward.value, 2),
        "energy_penalty": round(reward.energy_penalty, 2),
        "water_penalty": round(reward.water_penalty, 2),
        "stress_penalty": round(reward.stress_penalty, 2),
        "waste_penalty": round(reward.waste_penalty, 2),
        "healthy_reward": round(reward.healthy_reward, 2)
    }
        
    return (
        env,
        gen_obs_text(obs.model_dump(), env),
        round(total_reward, 2),
        json.dumps(reward_dict, indent=2),
        step_count,
        status,
        generate_plot(env)
    )

with gr.Blocks(title="Smart Irrigation Simulator") as demo:
    gr.Markdown("# Smart Irrigation Environment")
    gr.Markdown("Manually manage irrigation across agricultural plots. Keep soil moisture between 0.3 and 0.8 to maximize healthy rewards while minimizing water, energy, and waste penalties.")
    
    env_state = gr.State(None)
    total_reward_state = gr.State(0.0)
    step_count_state = gr.State(0)
    
    with gr.Row():
        task_dropdown = gr.Dropdown(["task1_easy", "task2_medium", "task3_hard"], value="task1_easy", label="Task Difficulty")
        reset_btn = gr.Button("Reset Environment", variant="primary")
        
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Status")
            status_display = gr.Textbox(label="Message", interactive=False)
            obs_display = gr.Markdown("Initialize to see observation.")
            
        with gr.Column():
            gr.Markdown("### Dashboard")
            reward_display = gr.Number(label="Total Cumulative Reward", value=0.0, interactive=False)
            step_reward_display = gr.Code(label="Last Step Rewards", language="json", interactive=False)
            plot_display = gr.Plot(label="Soil Moisture History")
            
    gr.Markdown("### Actions")
    gr.Markdown("Input the number of units to irrigate per plot (0 or more) this step. Max total irrigation across all plots corresponds to the current Pump Capacity and pump rules.")
    
    # We create inputs for up to 5 plots (max in hard task)
    action_inputs = []
    with gr.Row():
        for i in range(5):
            inp = gr.Number(label=f"Plot {i+1} Irrigation", value=0, precision=0, minimum=0)
            action_inputs.append(inp)
            
    step_btn = gr.Button("Step (Advance 6 Hours)", variant="secondary")
    
    reset_btn.click(
        reset_ui,
        inputs=[task_dropdown],
        outputs=[env_state, obs_display, total_reward_state, step_reward_display, step_count_state, status_display, plot_display]
    )
    
    step_btn.click(
        step_ui,
        inputs=[env_state, total_reward_state, step_count_state] + action_inputs,
        outputs=[env_state, obs_display, reward_display, step_reward_display, step_count_state, status_display, plot_display]
    )

if __name__ == "__main__":
    demo.launch()
