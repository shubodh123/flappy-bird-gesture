import pygame
import cv2
import mediapipe as mp
import joblib
import random
import numpy as np  # Add this line with other imports
import warnings
from sklearn.exceptions import DataConversionWarning

# Initialize and suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
pygame.init()

# Game constants
WIDTH, HEIGHT = 400, 600
GRAVITY = 0.5
JUMP_STRENGTH = -10
FALL_STRENGTH = 8
INITIAL_PIPE_GAP = 220
MIN_PIPE_GAP = 130
PIPE_WIDTH = 70
PIPE_SPEED = 3
DIFFICULTY_INCREASE = 0.997
PIPE_SPAWN_DISTANCE = 300

# Setup display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Eagle - Gesture Control")

# Colors
BLUE = (135, 206, 235)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)

# Fonts
font = pygame.font.Font(None, 36)
large_font = pygame.font.Font(None, 72)

# Initialize camera
cap = cv2.VideoCapture(0)

# Game state
game_running = False
game_over = False
pipes = []
score = 0
high_score = 0
current_pipe_gap = INITIAL_PIPE_GAP

# [ADD THESE NEW VARIABLES RIGHT HERE]
# Global display variables
screen = pygame.display.set_mode((WIDTH, HEIGHT))
fullscreen = False
show_camera = False
# [END OF NEW VARIABLES]

# Load eagle image
try:
    eagle_img = pygame.image.load('eagle.png').convert_alpha()
    eagle_img = pygame.transform.scale(eagle_img, (50, 40))
except:
    eagle_img = None

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Load gesture model
try:
    model = joblib.load("gesture_model.pkl")
    label_mapping = joblib.load("label_mapping.pkl")
except:
    print("Error loading model files!")
    exit()


class Eagle:
    def __init__(self):

        self.x = WIDTH // 4
        self.y = HEIGHT // 2
        self.velocity = 0
        self.width = 50 if eagle_img else 30
        self.height = 40 if eagle_img else 30
            # Add these hitbox properties:
        self.hitbox_width = 25  # Smaller than visual width
        self.hitbox_height = 20  # Smaller than visual height

    def update(self):
        self.y += self.velocity
        # Keep within screen bounds
        self.y = max(self.hitbox_height // 2, min(self.y, HEIGHT - self.hitbox_height // 2))

    def draw(self, surface):
        if eagle_img:
            angle = min(max(self.velocity * -2, -25), 25)
            rotated_eagle = pygame.transform.rotate(eagle_img, angle)
            surface.blit(rotated_eagle, (self.x - rotated_eagle.get_width() // 2,
                                             self.y - rotated_eagle.get_height() // 2))
        else:
            pygame.draw.circle(surface, ORANGE, (int(self.x), int(self.y)), 15)

        # pygame.draw.rect(surface, (255, 0, 0), (
        #     self.x - self.hitbox_width // 2,
        #     self.y - self.hitbox_height // 2,
        #     self.hitbox_width,
        #     self.hitbox_height
        # ), 2)


eagle = Eagle()


def generate_pipe():
    global current_pipe_gap
    max_height = HEIGHT - current_pipe_gap - 50
    height = random.randint(50, int(max_height))
    return {
        'x': WIDTH,
        'top': height,
        'bottom': height + current_pipe_gap,
        'passed': False
    }

def move_pipes():
    global pipes, score, current_pipe_gap

    for pipe in pipes:
        pipe['x'] -= PIPE_SPEED

        # Score update
        if not pipe['passed'] and pipe['x'] + PIPE_WIDTH < eagle.x:
            pipe['passed'] = True
            score += 1

            # Increase difficulty
            if current_pipe_gap > MIN_PIPE_GAP:
                current_pipe_gap = max(MIN_PIPE_GAP, current_pipe_gap * DIFFICULTY_INCREASE)

    # Remove pipes that have gone off screen
    pipes = [pipe for pipe in pipes if pipe['x'] + PIPE_WIDTH > 0]

    # Add new pipe if last one is far enough left
    if len(pipes) == 0 or pipes[-1]['x'] < WIDTH - PIPE_SPAWN_DISTANCE:
        pipes.append(generate_pipe())



def draw_pipes():
    for pipe in pipes:
        pygame.draw.rect(screen, GREEN, (pipe['x'], 0, PIPE_WIDTH, pipe['top']))
        pygame.draw.rect(screen, GREEN, (pipe['x'], pipe['bottom'], PIPE_WIDTH, HEIGHT - pipe['bottom']))
    # Debug: draw outlines for hitboxes
    pygame.draw.rect(screen, (255, 0, 0), (pipe['x'], 0, PIPE_WIDTH, pipe['top']), 2)
    pygame.draw.rect(screen, (255, 0, 0), (pipe['x'], pipe['bottom'], PIPE_WIDTH, HEIGHT - pipe['bottom']), 2)


def check_collision():
    global game_over, high_score

    # Screen boundaries check
    if eagle.y <= eagle.hitbox_height // 2 or eagle.y >= HEIGHT - eagle.hitbox_height // 2:
        game_over = True
        high_score = max(high_score, score)
        return

    # Eagle's collision rectangle
    eagle_rect = pygame.Rect(
        eagle.x - eagle.hitbox_width // 2,
        eagle.y - eagle.hitbox_height // 2,
        eagle.hitbox_width,
        eagle.hitbox_height
    )

    # Only check pipes that are near the eagle
    for pipe in pipes:
        # Only check for horizontal overlap
        if eagle.x + eagle.hitbox_width // 2 > pipe['x'] and eagle.x - eagle.hitbox_width // 2 < pipe['x'] + PIPE_WIDTH:
            top_pipe = pygame.Rect(pipe['x'], 0, PIPE_WIDTH, pipe['top'])
            bottom_pipe = pygame.Rect(pipe['x'], pipe['bottom'], PIPE_WIDTH, HEIGHT - pipe['bottom'])

            if eagle_rect.colliderect(top_pipe) or eagle_rect.colliderect(bottom_pipe):
                game_over = True
                high_score = max(high_score, score)
                return


def show_start_screen():
    screen.fill(BLUE)
    title = large_font.render("Flappy Eagle", True, BLACK)
    instruction = font.render("Show OPEN PALM to start", True, BLACK)

    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 3))
    screen.blit(instruction, (WIDTH // 2 - instruction.get_width() // 2, HEIGHT // 2))
    pygame.display.update()

# [ADD THIS NEW FUNCTION RIGHT HERE]
def show_camera_feed(frame, position, size):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, size)
    frame = np.rot90(frame)
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, position)
# [END OF NEW FUNCTION]

def show_game_over():
    screen.fill(BLUE)
    game_over_text = large_font.render("Game Over", True, BLACK)
    score_text = font.render(f"Score: {score}", True, BLACK)
    high_score_text = font.render(f"High Score: {high_score}", True, BLACK)
    restart_text = font.render("Palm: Restart", True, BLACK)
    quit_text = font.render("Fist: Quit", True, BLACK)

    screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 4))
    screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 3 + 50))
    screen.blit(high_score_text, (WIDTH // 2 - high_score_text.get_width() // 2, HEIGHT // 3 + 100))
    screen.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 2 + 100))
    # [ADD THESE LINES TO THE EXISTING FUNCTION]
    # Add the quit button
    quit_button = draw_quit_button()
    pygame.display.update()
    return quit_button  # Return the button rect for click detection
    # [END OF ADDITION]
    screen.blit(quit_text, (WIDTH // 2 - quit_text.get_width() // 2, HEIGHT // 2 + 150))
    pygame.display.update()
    # # [ADD THESE LINES TO THE EXISTING FUNCTION]
    # # Add the quit button
    # quit_button = draw_quit_button()
    # pygame.display.update()
    # return quit_button  # Return the button rect for click detection
    # # [END OF ADDITION]


# [ADD THIS NEW FUNCTION RIGHT AFTER show_game_over()]
def draw_quit_button():
    button_rect = pygame.Rect(WIDTH//2 - 100, HEIGHT//2 + 180, 200, 50)
    pygame.draw.rect(screen, (255, 0, 0), button_rect)  # Red button
    quit_text = font.render("QUIT", True, BLACK)
    screen.blit(quit_text, (WIDTH//2 - quit_text.get_width()//2, HEIGHT//2 + 195))
    return button_rect
# [END OF NEW FUNCTION]

def reset_game():
    global eagle, pipes, score, game_running, game_over, current_pipe_gap
    eagle.y = HEIGHT // 2
    eagle.velocity = 0
    pipes = []
    score = 0
    game_running = True
    game_over = False
    current_pipe_gap = INITIAL_PIPE_GAP
    pipes.append(generate_pipe())
    pipes.append(generate_pipe())
    pipes[-1]['x'] += PIPE_SPAWN_DISTANCE  # Make the second pipe come after the first



def process_gesture(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for point in hand_landmarks.landmark:
                landmarks.extend([point.x, point.y, point.z])

        if landmarks:
            prediction = model.predict([landmarks])[0]
            return [k for k, v in label_mapping.items() if v == prediction][0]
    return None


def main_loop():
    global screen, fullscreen, show_camera  # Add this line
    global game_running, game_over

    clock = pygame.time.Clock()

    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            # [ADD THIS NEW EVENT HANDLER RIGHT HERE]
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:  # Toggle fullscreen with F key
                    fullscreen = not fullscreen
                    show_camera = fullscreen  # Only show camera in fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
                    else:
                        screen = pygame.display.set_mode((WIDTH, HEIGHT))
            # [END OF NEW EVENT HANDLER]

        # Process camera frame
        ret, frame = cap.read()
        if not ret:
            continue

        gesture = process_gesture(frame)

        # Game state management
        if not game_running and not game_over:
            show_start_screen()
            if gesture == "start":
                reset_game()
        elif game_over:
            # [REPLACE THE ENTIRE game_over SECTION WITH THIS]
            quit_button = show_game_over()  # Now returns the button rect

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if quit_button.collidepoint(event.pos):
                        return False  # Quit if button clicked

            gesture = process_gesture(frame)
            if gesture == "start":
                reset_game()
            elif gesture == "still":
                return True  # Also still allow fist gesture to quit
            # [END OF REPLACEMENT SECTION]
            # # elif game_over:
            # show_game_over()
            # if gesture == "start":
            #     reset_game()
            # elif gesture == "still":  # This is the fist gesture
            #     return True  # Quit game
            # # show_game_over()
            # # if gesture == "start":
            # #     reset_game()
            # # elif gesture == "still":
            # #     return True  # Quit game
        else:
            # Gameplay controls
            if gesture == "up":
                eagle.velocity = JUMP_STRENGTH
            elif gesture == "down":
                eagle.velocity = FALL_STRENGTH
            elif gesture == "still":  # Fist gesture - could add pause functionality
                pass  # Or add pause feature here
            else:
                eagle.velocity += GRAVITY
            # Gameplay controls
            # if gesture == "up":
            #     eagle.velocity = JUMP_STRENGTH
            # elif gesture == "down":
            #     eagle.velocity = FALL_STRENGTH
            # else:
            #     eagle.velocity += GRAVITY

            # Update game state
            eagle.update()
            move_pipes()
            check_collision()

            # Draw everything
            screen.fill(BLUE)
            draw_pipes()
            eagle.draw(screen)

            # Display score
            score_text = font.render(f"Score: {score}", True, BLACK)
            screen.blit(score_text, (10, 10))

            # [ADD THIS CAMERA DISPLAY RIGHT HERE]
            # Show camera feed only in fullscreen mode
            if show_camera and ret:
                show_camera_feed(frame, (10, HEIGHT - 110), (100, 100))
            # [END OF CAMERA DISPLAY]

            pygame.display.update()

        clock.tick(60)


def main():
    try:
        playing = True
        while playing:
            playing = main_loop()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    main()