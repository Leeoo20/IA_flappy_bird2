import pygame
import random
import neat







# Initialisation de Pygame
pygame.init()

# Création de la fenêtre de jeu
screen_width = 400
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Flappy Bird")

# Définition des couleurs
white = (255, 255, 255)
black = (0, 0, 0)
yellow = (255,255,0)
red = (255,0,0)


# Définition de la taille de l'oiseau
bird_size = 30

# Définition de la taille des tuyaux
pipe_width = 50
pipe_gap = 200

# Définition de la vitesse de déplacement des tuyaux
pipe_speed = 0.2

# Définition de la gravité
gravity = 0.02


last_updatedifficulty_time = 0
jump_size = -1.7

        
        
# Définition de la classe Bird
class Bird:
    def __init__(self):
        self.x = 50 
        self.y = screen_height/2
        self.vy = 0
        self.size = bird_size
        self.color  = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

    def jump(self):
        self.vy = jump_size

 

    def move(self):
        self.y += self.vy
        self.vy += gravity

    def draw(self) : 
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.size, self.size))
        

    def check_boundaries(self):
        if self.y < 0:
            return True
        elif self.y + self.size > screen_height:
             return True
        return False
    
    def check_pipes(self, pipes) : 
         # Vérification si l'oiseau touche un tuyau
        for pipe in pipes:
            if self.x + bird_size > pipe.x and self.x < pipe.x + pipe_width:
                if self.y < pipe.height or self.y + bird_size > pipe.height + pipe_gap:
                    return True
        return False
        

# Définition de la classe Pipe
class Pipe:
    def __init__(self):
        self.x = screen_width
        self.height = random.randint(50, screen_height - pipe_gap - 50)
        self.color = white

    def move(self):
        self.x -= pipe_speed

    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, 0, pipe_width, self.height))
        pygame.draw.rect(screen, self.color, (self.x, self.height + pipe_gap, pipe_width, screen_height))
        
        


def main(genomes, config) :
    # Initialisation de l'oiseau et des tuyaux
    nets = []
    ge =[]
    birds = []
    

    
    for _,g in genomes : 
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird())
        g.fitness = 0
        ge.append(g)
        
    
    pipes = [Pipe()]



    
  
    # Boucle de jeu
    while True:
        # Vérification des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
           

        #Difficulte adaptable
        global last_updatedifficulty_time
        global pipe_speed
        current_time = pygame.time.get_ticks()
        if current_time - last_updatedifficulty_time >=10000 and pipe_speed < 0.5 :
            global gravity
            global jump_size
            last_updatedifficulty_time = current_time
            pipe_speed += 0.1

    
        #Deplacement des oiseaux avec IA 
        pipe_ind = 0
        if len(birds) > 0 :
            if(len(pipes) > 0 and birds[0].x > pipes[0].x + pipe_width) : 
                pipe_ind =1
        else :
            break
        
        pipes[pipe_ind].color = red


                
        for x, bird in enumerate(birds) : 
            ge[x].fitness += 0.1
            
            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - (pipes[pipe_ind].height + pipe_gap))))
            
            if output[0] >0.5  : 
                bird.jump()


            
        # Déplacement des oiseaus
        for bird in birds : 
            bird.move()
            
      
        
        #RECOMPENSE
        for x,bird in enumerate(birds) :
            if(bird.check_pipes(pipes)) :
                ge[x].fitness -= 1
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)
                
        
        #Si dépasse écran :
        for x,bird in enumerate(birds) : 
            if(bird.check_boundaries()) : 
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)
                
                

        # Déplacement des tuyaux
        for pipe in pipes:
            pipe.move()

        # Vérification si un nouveau tuyau doit être ajouté
        if pipes[-1].x < screen_width - 200:
            for g in ge : 
                g.fitness += 5
            pipes.append(Pipe())

        # Vérification si un tuyau doit être supprimé
        if pipes[0].x < -pipe_width:
            pipes.pop(0)

        
        
        
        
        # Dessin de l'écran
        screen.fill(black)
        
        for bird in birds : 
            bird.draw()
            
        for pipe in pipes:
            pipe.draw()
        

        pygame.display.update()
 
config = None
    
def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    

    p = neat.Population(config) #Population de base

    p.add_reporter(neat.StdOutReporter(True)) #stats
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(main,50)





run("./config-feedforward.txt")
    
        

        



