welcome today we're learning complex topics in a simple way we're doing

0:05

Frontline reinforcement learning hands on in the changes by the end of this

0:11

video you'll know Jimmer how it works its architecture and the code for it

0:17

does that sound good because you're here I assume that you roughly know what Jimmer is either

0:24

way I made a video about it that's like a non-technical overview the idea the

0:29

concept behind

0:35

it I showed there how dreams look how reconstruction looks so it's kind of

0:41

cool but as a quick summary dreamer is one of the best reinforcement learning

0:47

algorithms and it uses a world model so classic reinforcement learning plays the

0:53

game to learn the behavior but dreamer plays the game to learn the a word model

1:01

to understand what's happening how the world works and based on this model it

1:07

learns the behavior so it's slightly more complex but slightly I mean it's much more

1:14

complex but no worries because I'm here for you I wrote my own implementation of

1:19

it which you'll appreciate the clarity of but that's later forget about the

1:24

code for now first we have to learn the architecture we're starting with the publication and then their diagrams then

1:31

we'll get to my diagrams which are much better of course because otherwise I wouldn't make them and then we'll get to

1:38

the code is that clear my students any questions good let's

## Paper

1:45

begin so here we are mastering diverse domains through World

1:52

models and first of all you cannot be afraid of Publications you cannot be

1:57

afraid of reading the papers they are written by the people for the

2:04

people but the people AR always willing to read them we're lucky that this paper

2:11

is actually pretty approachable like look at their language it's not too complicated it's very pleasant plus it's

2:19

only 11 pages the rest are just like references and additions so it's it's

2:25

quite short but okay let's read the abstract together they right even though

2:31

there exist some specific Solutions creating a general Universal algorithm

2:38

has been a fundamental challenge so we present dreamer V3 which on one

2:45

configuration solves over 150 tasks and

2:51

got diamonds in Minecraft that's it it was not that easy

2:56

in my video I showed you that the the D in Minecraft I kind of cheated but let's

3:02

give him that that's fine they brag about the performance of course much better po you should know this if you're

3:10

learning reinforcement learning um dqn you should know this mzero you should

3:15

know what is conceptually but that's that's beyond the point they brag about the performance okay uh then we get to

3:22

the introduction and the introduction is just a longer version of the

3:28

abstract like I recom you watch it if you're interested for the background for The Vibes for the like industry info but

3:36

it's really nothing new uh we go further and this is what we care about we want

## Official diagrams

3:42

to study the architecture training process of dreamer

3:47

remember three steps to training dreamer gather data based on this data train the world

3:56

model and based on this model train the behavior

4:01

okay okay now let's read this let's see what they wrote here what can we read from this the word model encodes sensory

4:10

inputs into discrete representation Z so here encoder encodes into

4:16

Z that are predicted by a sequence model with recurrent State

4:22

H so this disc representation is also created by the sequence model

4:30

given actions a hold on a second okay they show that

4:38

this takes action but it's not just action the diagram shows that it also

4:44

takes previous recurrent State and previous discrete representation that starts to show you that the diagram is a

4:51

bit too imprecise but let's go further the inputs are reconstructed to shape

4:57

the representations so like this is justification of

5:04

existence of the decoder but I explain it later okay don't worry about it don't worry if you

5:10

understand nothing from this the actor and critic predict actions a and values

5:17

V and learn from trajectories of abstract representations predicted by the war

5:22

model so this is this whole part in one sentence I'm not going to analyze this

5:28

diagram I'm just going to reference it later as we go on because you know

5:34

learning from this is not a great idea what's better is when we go

5:39

down here we have more details of the word model here we have every neural net

5:45

listed with inputs and outputs so that's that's much more

5:51

precise that's much better okay let's go through this so we have the sequence

5:56

model that takes in previous recurrent State preview discrete

6:03

representation and previous action okay this matches our diagram

6:11

here and it outputs the recurrent State okay the encoder takes in recurrent

6:20

State and observation to produce the discrete [Music]

6:26

representation Dynamics predictor takes in the recurrent State and outputs

6:33

this representation hat Z hat okay like I don't feel like they

6:40

explain in the paper why why does it look like this why do you predict the same thing both ways I'll explain it but

6:48

like I'm just like Curious like I wouldn't understand it if I didn't read the code but it's fine reward predictor

6:56

predicts rewards based on recurrence State and this C

7:01

representation and the same inputs continue predictor and the coder this they mark it as recurrent State space

7:08

model but like it doesn't matter it doesn't make us any difference like I

7:14

consider all these nets the world model that we're training in this step here and as you probably noticed the Dynamics

7:22

predictor is like a new thing that exists like between these

7:29

layers here here here's a sequence model here should be Dynamics predictor that

7:34

creates this representation plus also an coder and decoder use their recurrent

7:40

State and they don't show it on the diagram why okay also they show you like

7:48

two types of observations here which is image observation and just Vector numerical observation in practice we use

7:55

either one of those it could be both but it's usually one of those so so let's just focus on images okay reading the

8:03

whole paper I like as I was first learning it I read the read the thing analyze this and I wanted to make my own

## My first diagram

8:10

better diagram and I made this which much better fit the publication this is

8:18

just me studying it and creating the most faithful representation the most faithful diagram that I could

8:25

make so let's study this for now later we'll get a better diagram that I made

8:30

based on like my code and I changed some names it will be better but for now let's study this because we're trying to

8:37

understand the publication okay we're starting with image

8:42

observation we pass it to the encoder

8:48

alongside the recurrent state right to the encoder to create

8:55

discrete representation but here we have a sequence model with recurrent state that

9:02

we pass to the Dynamics predictor that creates the same representation I mean

9:09

zad this time and question number one before we go further is why is this

9:15

discrete first of all this is one hot and coded Matrix meaning only one value

9:23

in a row can be one the rest are zeros that are all zeros here and the Matrix

9:30

is 32x32 it's variable but they used in the publication 32x32

9:35

but why is it discret why is it like this why not just have like a bunch of

9:41

numbers and values here for the answer I recommend Adan mayor's video actually

9:47

the last video he made seven months ago I mean the title is kind of mysterious

9:53

uh doesn't tell you anything but look at the thumbnail do you recogn nice the

10:00

[Music] diagram right so he's he made a video I

10:06

mean he studied this he exactly answers why is it discreet which long story

10:11

short it performs better but why is it like this and what's happening why did

10:17

they use this it's a great video Let's just say that thisr is for better performance question number two is why

10:24

do we predict this both ways from the Dynamics predictor and from the encoder

10:31

okay and it's because we want to learn the behavior let's look at the diagram

10:38

we want to learn the behavior without access to the

10:43

observations so for the world model to know what's happening to create the roll out to have

10:50

step by step what's happening without actual access to the game we need the Dynamics predictor here and and colder

10:59

is only the guide for the Dynamics predictor so we

11:05

create the encol representation this representation and it's a guide for for

11:12

Dynamics predictor because Dynamics predictor has access to recurrent state

11:18

only only and coder has the same recurrent State plus the observation the actual

11:25

observation so encoder here is better informed the representation it will

11:32

create will be better informed it will be a better representation but it's a guide for

11:39

Dynamics predictor so we can do the roll out we can Lear the behavior without

11:44

using the incoder at all because we will not have observations okay

11:51

so I'm made a complete mess here but um these two Nets are like cooperating they

11:59

need too much this is student this is teacher okay but it will be even clearer later

12:07

on and one more thing recurrent State I think of it that it encodes the past so

12:15

encoder based on the past and based on what we currently see

12:22

represent the present but Dynamics predictor is just like based on the past

12:29

figure out what's the present State I hope that it's somewhat clear so

12:35

far we'll reiterate on this in the future diagram the whole pipeline will

12:40

repeat the whole thing so no worry but for now we have the descrip

12:45

representation that we take alongside recurrent State and this this is a full

12:53

representation of the state of the world I call L full State because this

13:01

is like the past of the time data what happened recently and was the present

13:08

State that's why based on this together reward predictor and continue predictor

13:14

predict you know what's happening was the reward was the um was the continue

13:20

value meaning like did the episode end or not and was the chance for it and

13:27

then we get the decoder okay that we construct that based on this compressed

13:34

representation this full State tries to

13:40

reconstruct what it saw originally the question is why did the coder why do we

13:46

need it because I will tell you right away that we don't use the Reconstruction to anything like it's not

13:54

useful like directly to us like I made a video for it I made visualizations to like show you how well it works but it's

14:01

not like practical directly because rewards we need rewards to learn in the

14:08

imagination What Happens continue also but this is not needed because actor

14:14

makes decisions based on the full state so what's happening here in the paper

14:19

they wrote A mysterious sentence that the inputs are reconstructed to shape the

14:26

representations and this justifies the existence of the

14:33

decoder okay think of it like this the observation is

14:38

compressed to this discrete representation and this recurrent state

14:43

so the whole state of the world is compressed vastly compressed and this

14:49

compression can go too far and the coder applies pressure on the whole system to

14:56

be able to reconstruct the original observation you know with some with some

15:02

losses of course but to be able to reconstruct it based on these

15:07

representations that's why they use the decoder to shave their representations so they're more informative and

15:14

apparently the pressure from reward predictor and continue predictor is not enough because like apparently these

15:22

representations need to convey something useful for the reward predictor to work

15:27

well but apparently what wasn't enough I don't know it's kind of an interesting test to run what happens if you delete

15:33

the decoder but they use it so we use it and then we go to the next time step if

15:41

the recurrent state was initialized because there is no past there was not like nothing to convey in this recurent

15:48

state but here we already feed the the previous full State the action we have

15:55

new recuring state new representation new OB obervation because we kind of like go to the sequence of

16:03

our collected experiences and it goes on and on I hope

16:10

it's getting pretty clear right now how the whole thing works but this was based

16:15

on the publication I studied the publication here but next diagram is my

16:21

improved diagram that's based on my code with slightly changed vocab so you'll

16:27

have to get used to it but it's really much better if you understand my next diagram you'll have much better time

16:35

understanding the code also this diagram introduces the behavior learning that we're missing here but it it will be

16:41

really easy after you understand the world model training all righty welcome after a small break I know that we're

## My final diagram (WMT)

16:48

going to like the same thing over and over but there are always some changes and also I'm not sure what pace to have

16:55

right like it's my first tutorial like all of this is extremely easy for me because I spent like months on this okay

17:04

but for you like I don't know I'm trying to be toor with the knowledge I convey

17:09

so you know every feedback is welcome for the future but okay let's get to the diagram and you can already see the

17:17

change is being made hopefully I colorcoded the variables and Nets uh in

17:23

different colors first of all this is an important change that we split the en

17:29

from previous diagrams into two separate Nets I was analyzing other

17:34

implementations and they always do it like this and I personally also like this way

17:41

of doing things because encoder becomes like the choke point the the processing block between

17:51

observations and like representation of this observation kind of that's Universal

17:58

because observation can have many different forms it can be Vector values

18:04

it can be image maybe you can do something crazy and connect many types of different observations and somehow

18:12

process them so this is like like a point where we process the observation and standardize the form okay because in

18:21

COD observation I can set like a constant value like whatever the size of this is I can have a constant value here

18:29

it's always like compatible with with the input here so I just like the split we'll be doing it like this from now on

18:35

and thanks to the setup we can have a separate name for this net here that produces then this

18:42

representation so I called it posterior net because it's the network that creates the posterior

18:49

representation go with me for now and we can call this prior net and

18:55

it has like like from the name itself you can figure out that these two Nets

19:02

are connected that this is prior this is what we think the state is without

19:09

seeing the actual observation and this is posterior this is are like like

19:14

additionally informed belief about the current state so we have two different

19:20

Nets prior net and posterior net that produce two different variables prior

19:26

and posterior I also index it with with zero and these two things we call latent

19:34

States this is a latent State and this is a latent State people we call it like in other

19:40

implementations they will call this these two stochastic and this is

19:46

deterministic Recon state is deterministic and it's just like like I get where the naming comes from because

19:54

this is like the past is determined uh and if the present represent ation is uh

19:59

like probability based because it comes from the distribution but that's not what I want to focus on while naming the

20:06

variables so I will always call this recurrent state that includs the past

20:12

and the present representation is the latent State latent state is when I like

20:18

don't distinguish which exactly version it is just latent state is the discute

20:23

representation this is the latent State the one H and coded vector and I also used 16 x 16 but it doesn't matter

20:30

because we can freely modify the sizes um so Laden State this is a laden state

20:37

in two versions that come from two different directions and recurrent State comes from recurrent model but this is

20:45

them because like we initialize it to all zeros so it's not processed by the

20:51

model by this network like I just initialized it like in the loop so it

20:56

doesn't really come from the model but later it will come from the model okay and we create the full state by

21:03

concatenating recurrent State and lat state but in this diagram it's also

21:09

explicit that we use posterior for this posterior because it's the more informed

21:16

more accurate latent state right so we can more accurately train all these nets

21:25

okay so we create a full State we pass it to to the Nets reward continue and

21:30

the coder right and this full State alongside an action goes to the

21:38

recurrent model to create the loop again and continue the process and here I have

21:44

a note because it's important that action comes from the

21:49

buffer full State comes from the posterior it's important for you to remember this that action here comes

21:56

from the buffer because we chain the World model by replaying sequences of

22:02

what happened of what actually happened so we can match our predictions to what

22:07

actually happened because in Behavior learning we'll have a recent model that

22:12

takes live action from the actor and full State creating with the

22:19

prior okay but that's foreshadowing don't worry about it for now so we were at this time step to create recurrence

22:25

State one we take full state zero and action zero right prior one posterior

22:32

one observation one full State one reward one continue one reconstruction

22:38

one and then the cycle continues and it goes until it reaches bch length steps

22:44

so we predetermine it's a like a parameter how long the sequences we

22:51

train on so this is how the word model training goes we instruct we like press pressure

22:59

we order the prior net to match the posterior and we order all these nets to

23:05

match what happened in reality when we were playing the game right and and

23:11

coder is changed just like in directly like we back

23:16

propagate through like all these nets so the en colder produces useful

23:22

representations now we passed the sequences I mean this is just example of

23:27

one sequence but there are like multiple sequences in parallel doesn't matter like we iterated on the word model now

23:34

it's time to advance to behavior learning to actor critic

23:40

training you see the lines here this is how we advance to The Next Step because

23:47

you remember the diagram From Here We Begin actor critic learning that's also

23:52

called Behavior learning with one observation that's encoded

23:59

and then we imagine just in this slate representation without access to the to

24:05

the game okay that's how that's how it's done that this full state is fully

24:12

grounded because it's created using the posterior well informed posterior and

24:20

this well-informed full state is passed on further as an initial

24:26

step for the imination roll out right I don't have to encode it again I don't

24:32

need new observation I can just connect the dots like I have already this representation here I don't need to

24:37

encode it again like these two are not separate I just nicely connected because

24:42

full state is already a full State I don't need observation okay so we pass

24:49

the full States further in my current code I pass every full state in theory I

24:54

believe you could just sample like sample a specifi number of these full States from all these sequences but for

## My final diagram (BT)

25:01

now I pass all the full States and we go further I colorcoded the Nets

25:09

differently to distinguish it oh I even wrote it here that this recurrent net is

25:15

it's like the same net that's here but it's not being trained at the moment

25:21

it's constant right we only train actor and critic here in this step so I just

25:28

Chang some colors we have the well informed full State here that's passed

25:33

to the actor because like this is actually like a new observation it's encoded but it's full observation full

25:41

state so like every net operates on on this full State that's why I cared about

25:46

connecting it into one one concept instead of having like recurrent and um

25:52

lat State separately so we have the well informed full State that's passed to the

25:57

actor to create an action and the same full state is passed to the recurrent model along with that action to create

26:05

the recurrent state in imagination this time right so this reent

26:12

state can be passed onto prior net that's trained like using the posterior

26:18

but now we have no posterior now we have just the prior and this recurrent State and this prior come together to form a

26:26

full State yellow this time because this was from the posterior this was well informed but this is just our

26:33

imagination full State and this fully imagined full state is passed to the

26:41

actor to get an action to the critic to get the value to the reward model to the continue model and they are not being

26:48

trained they are guides to actor and critic okay they are

26:55

guides and this action goes along with the full state to the

27:03

recurent model and we're in the same spot that I mentioned earlier that this is Rec model

27:11

but it takes live action from the actor and full state from

27:17

prior while it used to take action from the buffer and full State using

27:23

posterior right do the dots connect in your head please please say yes so we

27:30

have a new recurrent State new prior new full State new actions new

27:37

rewards new Rec State new prior this was like the Pinnacle of what

27:44

I was trying to teach you oh I even I even forgot because like here we don't

27:50

use the decoder I mean I wrote it here because we don't use the decoder unless

27:56

you want to show your audience how the dreams look okay but here the codor is not traditionally used I use it

28:03

just like for fun visualizations but um it's pretty much not existent here and

28:09

then continue it continue until it reaches imagination Horizon

28:15

steps that in practice both the dreamer and I use 15

28:23

okay now this was one roll out one roll out one trajectory but from

28:31

every full state from the world model training for every initial State here we

28:37

create separate roll out right and that's the whole training

28:48

pipeline let that sink in and that's the end of our Theory here how does this

28:54

diagram look now huh it was like too vague at best and misleading at

29:02

worst okay that we now we got settled then we

29:07

scroll down and then we get to the details to concrete losses to tricks

29:14

like 1% uniform mix that I will be showing you in the code already all the

29:21

losses here the equations this is Lambda values

29:27

calculation actor critic okay Advantage maximization Plus entropy term Sim log

29:34

and simx okay too hot oh

29:40

no that hurts because this actually doesn't work yet in my implementation but it's not necessary it's just like

29:47

it's like a small thing but kind of annoying I wrote it correctly everything's fine it just just doesn't

29:54

perform well like what do I tell you uh so some more

29:59

results yeah and then it's just then it's just that all righty oh my goodness

30:06

are you ready for the code brace yourself but before we get to

## Comparison of implementations

30:12

the code I wanted to start with this redit post dreamer V3 code is so hard to

30:18

read because it really well sums up my experience and many other people he

30:25

spent three months trying to understand the um the paper and the codee and after

30:31

3 months he's still overwhelmed I mean he was still overwhelmed and didn't know

30:37

what to do and you have to trust me that this code is really this is the original

30:44

g code and it's really complex they support so many different like

30:50

Frameworks and ways of doing things like here I remember that they like for

30:56

saving the metrix they support Json L whatever that is tensor

31:01

board and that's like industry standard uh expa One DB scope and like this is

31:09

the main script like can you can you understand easily what's happening here there's so much code and I have a good

31:16

example for you to illustrate this well this is for example how critic is

31:23

trained from the paper like before we get into much detail but just just know

31:28

that we minimize this is the loss we minimize this so we maximize the

31:34

probability of returning Lambda values based on the state okay and Lambda values are

31:42

calculated like this with this one equation where we take into account rewards

31:48

continues current evaluation from the critic and Lambda return from The Next

31:55

Step so to get the Lambda return of this step we can of like reference the next

32:00

value and the next and the next so pretty much the way to calculate it is to start from the from the very end

32:06

because uh the final value is given that is just evaluation of the critic and then we work backwards with on this

32:13

equation and let's look at functions that implement this Lambda return this

32:20

equation this is original GMA code like

32:28

[Music] okay they do something

32:35

here The indexes are quite complex I need to like I need to spend some time

32:41

on it then they iterate uh backwards then reverse the list again the variable

32:49

names aren't even like full rewards like how much would it cost them to call this

32:55

reward and we have to use the bootstrap so like the bootstrap is the is the last value the reference value and the

33:02

bootstrap shifts um constantly with each step

33:07

but why do why do we have last and bootstrap and dis is what a discount

33:13

like this is quite unnecessarily complicated let's look at sheer L version this is a nice Library I studied

33:20

their code a lot but let's look at the version it's the same thing with reverse

33:26

list we reverse again we Stu no we don't stack it we concatenate

33:33

it and they calculate the intermediate values why why it's like yeah this can

33:40

be done like in advanc without the list but like I don't know why I don't know

33:45

if it's quicker than than Michael that I will show you in a second the simple

33:50

dreamer they adjust indexes here they do the the same thing they stuck revers

33:59

again okay fine but then look at my

34:04

code I don't believe my code is slower if it's slower at all but it's most

34:10

importantly so much cleaner like I don't need to pass pass the

34:15

device I don't need it I just like inere the device from from rewards I don't

34:21

need to pass in Horizon length it's INF feared based on reward shape I create the buffer for Lambda value

34:28

set the bootstrap at the beginning to to the last value then go in reverse

34:35

calculate the equation just this is the same equation perfectly so you can easily understand

34:41

it and see how it works then we change the Boost STP and it works perfectly why

34:49

why why yeah okay so now you understand how um

34:55

roughly how like what was the idea behind my code what was the approach and I was pretty much inspired by this

35:01

because I was trying to learn it I was just like wait I can do so much cleaner and help so many other people so that's

## My repo

35:08

why I did let's close this let's see my actual code natural

35:14

Jer I called it I called it natural Jer because I want to emphasize that my G is

35:21

like a it's like a natural like he shows up he doesn't really fit I wanted to call him Underdog G or amateur dreamer

35:29

because he's not a professional I mean wait all these implementations are just like serious teams we academics we we

35:37

want to have performance we like do stuff I show up I just I just studied it

35:42

I don't know I'm not in the in the in the industry even I just like came here

35:47

I wrote clean dreamer because it's um this is what I do this is how I think

35:53

and this is pretty much these are pretty much other dreamers they use super macro lenses for anti-polar

36:03

anti-reflective laser microscopy telescopy Zoom my dreamer shows up and

36:09

it's just like I don't know I I shoot like I try to hit the middle things like

36:16

this like this is natural dreammer so you can check it out you can start it

36:22

start it I dare you but um let's get to the code now let's analyze the code I'll

36:27

show you my local version that I have here this is the main script uh okay I

36:34

already started making some um some changes so I'll quickly stash it

36:40

so we don't have any changes because I forgot to mention that I made it nice for you because there was a whole mess

36:47

with many different versions I had some attempts had some failures different repos so I just put it nicely for you

36:54

that this tutorial is made on my first Comet so if you want to have the same

36:59

code the exact same code just go to the first comet of this people okay and then

37:05

you can trace back the uh the history uh what did I change I already

37:11

have like I already see things to change I already uh like have some changes in mind I can make endless changes whenever

37:19

I go to the code I see like wait this can be done better so this will change for sure uh I need to add more features

37:26

for now only one environment is beaten and the car racing but you know let's get to the

## Code overview with main.py

37:33

code now we operate on config files that you see here on the on the side so all

37:40

the parameters are here on the side um the main script is enough for

37:49

you to understand what's happening how the training functions and I'll show you

37:55

how easy it is like with original GMA you see Main main.py and it's like what

38:02

is even happening you see my code you can just like go L by line and understand what's happening so I'll hope

38:09

you appreciate it so let's begin what happens when you run this code this

38:15

happens so this just runs out of the box um this is a classic if you don't know

38:22

this this is a classic equ will just like um pass the arguments from the common line I don't really use them but

38:28

they might appear in the in the future uh I might add more parameters here but

38:33

for now it's not necessary I just have one config does the default so I don't have to pass it even I just run the

38:39

script and and it works so I pass the arguments with one string for now the

38:46

config name I have one config so this just executes the main script with this

38:52

config so we start here this is the loop um I mean you saw that we print uh the

38:59

environment properties that's from here you can see the properties for later uh

39:06

channels height width action size action boundaries this is important for

39:11

continuous values because they actually differ I mean steering can go from minus to to plus this is gas from 0 to one and

39:18

break from 0 to one but that's that's for the future I don't know why I mention it but it's too late now I did I

39:24

mentioned it so uh let's start with the script we uh we have the config how it's

39:31

loaded I mean I'm not sure in what level of abstraction to speak I'm just trying

39:36

to like you know tell you what comes to my mind to give you like a like a feel for it

39:42

but but okay let's go with it we load the config um we load the yam file this

39:51

file as a python dictionary that we convert to aict

39:58

which means that we can like reference we can index the python dictionary with

40:04

attributes so for example when we have the config um config variable config to

40:10

get the seed because this is the the base level of of this config we can just go config do seed config do Seed where

40:19

is it um somewhere here config the SE and then to get B size we we go config

40:26

do Dreamer do B size like this and it's

40:31

very a clean way to work with it then we see everything I mean that's obviously

40:39

has to be done we want to eliminate um the effect of Randomness we have the we

40:45

want the runs to be reproducible uh so when we make a change we want to see the effect of this

40:51

specific change uh not just like random seeds because the the runs quite

40:57

different so like we always see everything remember about this uh then I

41:03

have like uh I figure out the folders and Run Names because here in config you can name your

41:09

run for example I can show you metrix files like all this are are like full

41:16

run names with the environment name and my my specified name in the config so

41:22

like just distinguish the run I mean I made a lot of the

41:27

and that's just like the past two months it doesn't include like a ton of other ones so yeah that took me a

41:35

while but then we figure out the paths for uh for the folders because we need

41:41

to have the folders created that here in the conf below the names Matrix plots

41:47

checkpoints and videos I automatically create these folders because otherwise you'd get an error because they have to

41:53

exist but I will not like include in the repo the videos because these are my local

41:59

files so I do this I sure that uh the folders exist with this function you can

42:06

read the details right I think I will get into like this is this is so simple

42:11

that um it's not crucial to understanding the algorithm I'm just like giving the background of like how

42:17

do I work with it and what's what's happening so that's that I think that's that should be pretty clear and and

42:24

these are called bases not file name but file name base because we'll have many

42:30

checkpoints and many video file names uh I just append like later on

42:37

I'll append the suix to them so checkpoints have the Run name plus 2K 4K

42:44

8K 10K stuff like this but okay I think I'm talking too much I think I'm giving you too much detail what do you

42:51

think uh let's go further uh then I create the environment this is the funny way to

42:57

create environments like Nobody Does it like this that we have two environments

43:02

okay think about this because I want to create the videos to inspect every

43:07

checkpoint how it behaves what's happening I have all the checkpoints and videos for every

43:13

checkpoint to like you know uh to like see what's happening and to record videos I need to have the random mode

43:20

past of LGB to the environment I mean if you didn't work with gymnasium these are

43:27

just industry standard uh environments for reinforcement learning originally

43:32

developed by open AI back in the day when they were doing a real um yeah so

43:39

anyway like you should probably know this like I will not explain it here because that's that's like super basic

43:44

to figure out how to how to work with it but okay uh I was saying that I have

43:51

two environments I need to pass the render mode to to record videos but

43:57

there's a chance like I cannot find definite information on the internet but I believe that's what Chad GP told

44:05

me that like they could skip some internal logic if I will not have the

44:12

render mode because um default render mode is nonone so I I cannot even render

44:19

the FL the frames uh but I want to render the frames for videos but I record videos

44:25

only on on a small m minority of um of the runs so I was just like okay

44:33

hopefully I I mean let's do it like this I created the base environment on which

44:38

I train so it's like as fast as possible because I train I don't have I don't need to render the the frames then I

44:45

have a special environment for for evaluation for recording the videos

44:51

and I just I just hope that it's really quicker this way because otherwise it

44:56

looks stupid I I I've never seen anyone do it like this it's all very weird very

45:03

very crude it's like a like a Gilla tactic of an amateur so I don't know I do it like this for now it might change

45:10

if you if you watching the future and and you see that it's it's gone then it was

45:16

stupid and then I get the environment properties in one line nicely I'll not

45:21

elaborate we just um called them uh the gymnasium API to to get the uh get the

45:30

properties that you saw here right these are the the the

45:35

properties that we pass uh in the future will need more we need to have uh I mean

45:41

for now I only have continuous uh action space but in the future I'll be probably

45:47

passing uh you know the type of the action space I'll support like normal um

45:53

non-image observations so this would change but like this anyway uh for

45:59

now then we have the dreamer we we instantiate the dreamer class and then

46:06

we pass the config Dreer remember so what dreamer will have access to like

46:12

this level of the config will be its base

46:17

level like it it will just call self. config B size so it's neatly organized

46:24

Nobody Does it like this I actually organize it very well in the in the GMA you can see that every subass every um

46:32

every neural net has its specific config nicely passed they don't like take internally

46:40

what they need from the config they just like they just have what I want them to have so in COD they will have only these

46:48

four that's nice I like it uh so that's config uh if we want to resume if the

46:53

flag is true here if we resume true then it will load the checkpoint specified

47:00

here this specifies the the suffix that we that that we want to create I mean that that we want to take with a

47:07

specified run name right okay quick break

47:16

because my thr is going to go crazy then we have environment interaction and this function is my

47:24

equivalent of like a uh you know on on the high level so you know the internal

47:30

internal logic is hidden underneath the function but this is my function to

47:35

gather new data remember the dreamer gathers data

47:41

change the world model on this data and change the behavior on the word model

47:49

okay so uh so this will be the function to to gather new data to play an

47:54

episode specified number of episod actually but it's usually one except the

48:00

the warm-up because this is be like this is the training Loop this is before the training Loop we play like um episodes

48:07

before start so these are like warm up episodes before we start training to have some data for the diversity we like

48:15

diversity yeah uh then we have the chaining Loop uh that's organized in two

48:21

Loops because we train the model here

48:27

Here We Gather new data and this ratio this replay ratio determines how often

48:34

we train versus Gathering data like I can straight up just uh for now it's 100

48:40

that's what they used in original dreammer uh I can just straight up increase it to to increase the

48:47

effectiveness of like like sample efficiency so we can learn more on the same number of data yes

48:57

so this is our training group here we quickly figure out um how many interations we have to make of this

49:04

based on how many grading steps we want and what's the replay ratio so we sample

49:11

data here we train the word model with the data

49:17

right we get the initial States I mean we return matric that's for the bagging that's that's only for me uh and initial

49:25

States so the the full states that will Kickstart the roll the roll out process

49:32

and then we change be the behavior using the initial States and the the world model and we return the metrix and

49:39

increase the number of gradient steps because we want to track them right for

49:44

later this executes if I mean if uh the

49:50

number of gring steps happens to be divisible by the checkpoint interval and we want to save checkpoints then we save

49:57

a checkpoint and every checkpoint is figuring out the suix um or saix I think

50:05

it's suffix uh I divided divide the gradient steps by a th000 and add the k for

50:12

shorter you know for shorter names because this will be like too long and you'll specify the suffix here

50:20

remember uh with the with the checkpoint to load then we save the checkpoint of

50:25

specified name of the base plus the suffix then we evaluate the score uh

50:32

with our weird evaluation environment uh and we save the

50:40

video right and this like in the evaluation mode we don't gather new data

50:48

so this is purely for evaluation this is not for Gathering data but we return the

50:53

score that we got and we printed on the console just to track the process of the

50:59

current stage of the of the process so that's that's the loop that runs 100

51:06

times I mean here this is unnecessary I already change it in my I have it in my

51:11

changes that that this two are unnecessary it will be like this but for now I cannot really do it because I

51:18

already committed it I cannot make changes so I'll push it later but the

51:26

functionality is the same so it's fine and then after we trained for aund

51:33

steps We Gather new data with this function and then if you want to save

51:39

the Matrix if the flag is here true it's true by default we get the base um base

51:46

info about the environment steps grading steps recent score from

51:53

the from this function and then we we connect the Matrix with the word

52:00

model Matrix and actor critic Matrix and then we save it to the CSV

52:06

file I can show you the file uh let's see something good that's the file big file of just

52:15

helpful metrics K loss actor loss CRI loss advantages just to for me to see

52:21

what's happening and like with this file I can do anything right I can open it in Excel

52:28

and make some make some graphs make something but then we also plot the Magics so I have a special function that

52:35

creates an HTML plot from these matrics for me to inspect and play around I can

52:41

show you the plot this is the plot I mean uh the scale is messed up so if we turn off this and this we see some

52:48

chases we want to see only total reward this is the run this is this is

52:54

what you saw on GitHub really nice training here I believe it's overtrained maybe it will get up with

53:02

time but I don't know I'm not convinced like still like 900 average reward

53:07

because this is averaged over um 10 um 10 steps 10 episodes so over 900

53:15

rewards is like Max on this environment so like this is solved like here it was already solved and solved here so we

53:22

don't need to train anymore this is just overtrained I believe so overall really good

53:29

run um and that's it that's the main script right like I I told you like this

53:37

is this is the callor ga data change the world change behavior

53:42

and and it's very clear how you would proceed like if you want to inspect how

53:48

the world model Changs just just follow the function and see what's happening

53:53

like it's obvious like you have the the top the main script and then you can Branch out with with all these other

54:00

like all the other like original dreamer I open the script and I'm just like

54:05

dude no idea I cannot tell you what's happening there so this is

54:11

nice I'm really proud of this as you see like if you really want to understand it

54:17

because that's how I used to um learn me person learning that like I started by

54:22

taking someone else's code and just deleting everything that that's unnecessary for me not like unnecessary

54:30

overall but like in the moment for learning like every line every line

54:37

matters like the metrix why would I need the metrix what I'm trying to understand the algorithm no metrix not this I don't

54:46

need checkpoints even delete the checkpoints what's happening here I

54:51

don't need the the files I don't need zero files I don't need this I don't need this I is I mean okay that's that's

54:57

a bit too much but you get the point like it's clear what's happening I

55:04

hope nice now we can proceed with more details I think we'll

55:10

get to the world model training then behavior training

55:16

then the neural networks just like a quick overview of what's happening so you know like I will not explain to you

## World Model Training

55:23

everything like I want you to know what's happening so if you want to understand something

55:29

like you know what you have to research all righty word model

55:35

training we'll go through this but for educational purposes I took the whole

55:41

function out into the jupit notebook so everything's the same except the self

55:48

keyboard has been exchanged for dreamer so it works but you know that we're

55:55

analyzing this which is in jupyter Notebook so our main Loop passes data to us the sample data

56:04

right we're getting the data and then we process it so we first pass the

56:10

observations from the buffer reshape them this way then we pass through the

56:16

en colder reshape it back why is it done this way let me I

56:23

mean first of all data observ

56:29

okay of course I had to run it first okay so this is the observations from the buffer this is batch size batch

56:37

length and observations right three channels 64x 64

56:43

sequences of 64 times steps and 32

56:48

sequences like this that are specified in the in the config so it can be

56:54

complicated but uh pretty much we do this reshaping trick because

57:00

encoder expects at most for dimensional

57:05

um input because it has it knows that there are three channels consisting of I

57:11

mean the image consisting of three three dimensions here but it has only one

57:17

like it has only capacity for the bash Dimension and we have two additional

57:22

Dimensions so we pretty much collapse this

57:28

together so uh let's see how how that

57:34

goes so I do it like this dot shape we collapse these two Dimensions together

57:42

so this number is 32 * 64 okay then we pass it to the and call

57:53

there oh shape shape to get this so we have the same

58:02

botch size times botch length but the the three last dimensions are just

58:08

collapsed into this representation this enced observation and this side is the

58:14

size is specified in the config right and call it op size okay this is the size and then

58:23

we reshape it back because we want to have it in uh in the original form so we

58:29

reshaped back 32 6424 this is the trick to just pass the

58:36

whole thing um with B size and Bash length through the encoder okay I hope

58:42

that's uh that's somewhat clear so we have the encoded observations then we

58:48

initialize uh prent State and later State and I have to immediately apologize because that's not what I

58:53

showed in the diagram uh because this went through like many

58:59

reworks I I I didn't check I forgot to check and that's the diagram I created

59:05

earlier stuff happened my mistake that um this is the fixed diagram so we don't

59:12

initialize this recurrent state with zeros we initialize recurrent State here

59:17

and Laten State we take our first action from the buffer to just get the new

59:23

recording State and now we create the roll out meaning that we don't use like the first the zero observation here so

59:33

we encoded here we encode all the observations but because of the setup we cannot ignore the the zero observation

59:41

so that's my bad I believe it can be because I had some problems like I will

59:46

not get into detail why why I had to rework it like this but I believe I could fix this in the future so this

59:51

might change but I'm sorry the diagram didn't match my call in the in the

59:57

previous version but that's that's the only change the rest the rest stays so we initialize the curent state

1:00:03

and L state with uh with zeros uh of the specific sizes and this length of that

1:00:10

action is just our bash size I don't know why is it like this I believe that

1:00:15

um it's because it's shorter this way but it's less clear so I'll change it in

1:00:21

the future comment but it should be like dreamer like self in the real code and

1:00:26

Jimmer config dob size like it'll be it'll be longer I don't like the length of this

1:00:33

line but but it's still fine it's still fine so this is just um B size we initialize them but we don't

1:00:40

have the sequence length here I bet like some people might ask about this because

1:00:46

this Rec State represents a singular time step then we'll we'll stack them

1:00:52

together and you know have the bges and sequences but this for now is just one

1:00:57

time step so it it only has the bad size dimension not bash length then we initialize the uh the

1:01:06

list and we make our roll out that we

1:01:11

all know and love recur state passed uh with these enise values and the action

1:01:19

right this we take uh Rec State and pass it to

1:01:25

the prior net we have prior here this is prior so the output of of this net but

1:01:30

we don't use it in World model training so we just need the logits for further

1:01:36

training uh we take posterior posterior logits right we only use the posterior only

1:01:43

posterior um and we pass with the posterior concatenated recurring State and ened

1:01:49

observation I like it this way because many people like pretty much everyone I

1:01:54

I checked is concatenating like it's it's passing these two uh or like universally if there are

1:02:02

like two things that have to connect in the net they pass the two things and then connect them internally in the net

1:02:09

but I but I like to treat it as as one entity that I connect and posterior net

1:02:15

just gets one input of specified size that's also why I connect the recur

1:02:22

State and Laten State into full State like I feel like it's easier to work

1:02:27

this way all right uh then we append to all these lists are for

1:02:35

outputs that we passed and then we change the previous state and previous Len

1:02:40

State then we stack them on the sequence length

1:02:47

dimension on the like time step Dimension because by default it's uh

1:02:52

it's zero so it would um it would stack them on on the First Dimension but that's not what we want so we pass it

1:03:00

and we create the full States from recurent States and posteriors here we just concatenate so we have one one

1:03:07

thing one full state that we can pass to other Nets so now it's time to train the

1:03:12

Nets decoder we pass to decoder just full States that's it but we do that

1:03:19

reshaping trick because um we also have to pass the five dimensional input

1:03:26

but decoder only takes four dimensionals so we do it like this and we treat the

1:03:31

decoder output as means of normal distributions of deviation one that's

1:03:39

how it's done uh we still just maximize the probability of

1:03:45

returning uh what we observed in the buffer right that's the loss that we all know and love minus log problem that we

1:03:54

return this and that's independent because like every distribution that comes from here

1:04:01

is uh is independent they shouldn't be influencing each other in any

1:04:06

way so then we have the reward distribution uh predict rewards from

1:04:12

Full States but this time reward predictor already returns that distribution that we can directly call

1:04:19

log proon and we just maximize the probability that we will return the

1:04:24

rewards from like ignoring the the the zero reward right because we don't have the

1:04:31

the full state for it and that's done and then this is a bit complicated uh how do

1:04:38

I explain it to you let's go to the paper where is mastering diverse domains through World

1:04:46

models so I mean I probably will not explain all this to you but how we train

1:04:52

the two Nets um I don't know why they distinguish into two separate losses I just uh add them together because it's

1:05:00

just one k loss for me but uh what they want to do is they

1:05:06

want to minimize the KL Divergence I

1:05:13

mean it's like you have to read about this

1:05:19

long story short we just like Kar Vin is a measure of how two different

1:05:25

distributions differ from each other okay so we just

1:05:30

take the RO loges we create um category called distributions from it uh from

1:05:36

them and then we want to match them together and how they do it I'm not fully con

1:05:43

like I wouldn't be able to like fully explain it to you but they calculate the

1:05:48

same thing that's um like these two losses here the equations are the same like the values will be the same

1:05:56

but the Dynamics loss stops the gradient here on the posterior like

1:06:02

output and we want to minimize the the the difference here but only like into

1:06:09

the prior will flow the the gradients and here representation loss

1:06:14

is the same thing but we stop the gradients here and they have different weights like what weights do I have it's

1:06:22

uh one for the prior for like for this Dynamics and 0 one for the second one so

1:06:28

we'll be changing the prior more to match the posterior but posterior is also like slightly matching the the

1:06:36

prior it's a bit complicated I'm not fully sure why it's done like this but

1:06:42

um that's how it's done and we also Max the whole thing with three nuts of one

1:06:48

and they do it so if these two distributions like match each other decently well that the loss is like like

1:06:56

smaller than one then we just max it meaning that no gradients will flow

1:07:02

because it's it's it's a constant right there's zero gradients we just focus on different Nets we

1:07:08

prioritize them so that's how I do it I create four distributions with lodges

1:07:14

from prior and detached from posterior and detached so I have four of

1:07:22

these then I calate the kale Divergence I I create the tensor with the threee NS that I Max

1:07:31

with prior loss and posterior loss I apply the weights I add them together

1:07:37

mean and that's my K loss okay I mean

1:07:43

that's as much detail as as I will give you right now because you know the topics

1:07:49

here are a bit uh are a bit tough so we have the Reconstruction loss the reward

1:07:54

loss the K loss then we add them all together right they calculated

1:08:00

separately like this I mean they have like the weird naming convention and

1:08:06

then they add them all together here it's uh it's pretty much the same thing I just like slightly reframe it uh so

1:08:13

that's the word model loss and if we use the continuation prediction that is not tested in my implementation so I haven't

1:08:20

used it before uh but I sted it it should work pretty pretty well but if we use it and the flag is what here if we

1:08:28

set it to True will add to the world model loss the continue predictor loss

1:08:34

from Full States calculate the loss of real data we know the drill with this

1:08:40

then we step the optimizer and I get my metrix uh I mean

1:08:45

I can explain the Kos shift for graphing because because of this Maxes here and

1:08:52

the weights here the K loss cannot go below 1.1 so it's always shifted and I

1:08:58

don't like the shift for my for my plots and the bugging so I just shift the uh

1:09:04

water model loss and K loss so it's like back to back to zero but it doesn't change the numerical values here in the

1:09:12

actual training Loop and that's it we return the initial stat so the full

1:09:18

States from here but we shap them because they like the sequences the the B size and Bash

1:09:26

length don't matter anymore we don't distinguish them we just have all the full States uh that we of course detach

1:09:33

we don't want any gradients there just in case and we return the Matrix right

1:09:38

this was our Jupiter notebook let's delete this uh so this is the actual function and we just return this stuff

1:09:47

hopefully you understood something at least okay please say yes please say

1:09:52

yes then we get to the behavior training so the only input here is

## Behavior Training

1:10:00

initial states that we call full State here uh like this because this is the

1:10:05

like one time step only uh that we get here like I wanted to pass only singular thing for the for

1:10:13

easier cleaner main um main function and

1:10:19

just a nice framing I like the the framing of having like passing only one thing but unfortunately uh for practical

1:10:26

reasons I have to split it into Rec State and Laden State because that's how

1:10:31

recur State works that I have it separately here I don't contain like one package because I'd have to split anyway

1:10:38

because uh recing state are is being passed to the recurrent network but

1:10:45

these like um other inputs are being processed by by just linear fully

1:10:50

connected Network so so unfortunately I have to split it but the rest works

1:10:55

uh as we remember from this initial State we created a roll out of

1:11:02

imagination Horizon steps we just get the action I me let's

1:11:08

get to the to the diagram that's slightly updated that this full state from uh World model training is being

1:11:15

treated as the zero full State and is being passed to the actor and recurrent

1:11:21

model right this full state is two the actor and

1:11:29

recurrent and recurrent model from this we get the

1:11:36

prior right we concatenate it uh to have the full States uh before even appending

1:11:43

we append them we stack them and remember that when we operate on the 10

1:11:48

Source we would ignore the zero log problem like this we take the whole

1:11:54

batch and and then ignore the zero time step okay but here because I do it here

1:12:02

I don't even stack it and I like this is still the list in this moment that's why

1:12:07

I index it like this so on the tensor is the second dimension but on the list is list is

1:12:14

onedimensional right so don't mix this up and but okay then we pass the full

1:12:20

states to the reward predictor and remember that reward predictor Returns the distribution so

1:12:27

this just takes the mean of the distribution to get the actual reports so like the the the reward that has the

1:12:34

highest probability according to the reward predictor according to this distribution that it returns and we also

1:12:40

ignore the last time step is because we will not use this reward remember this

1:12:46

we need the rewards but for the last evaluation we don't need it because we

1:12:52

uh we have to have the bootstrap so we don't need the last reward so I ignore it but we need all the values from all

1:12:59

the full States and we also take the mean because that's what create Returns

1:13:05

the distribution and then we get the continues that already incorporate the

1:13:10

discount so if you're looking for like the gamma Factor where's the discount so this is how we would pass the the full

1:13:17

state to the continue predictor but we don't use it right now I have the tensor of just discount values thinking that

1:13:25

continu is one okay and then we pass it to Lambda values that we already went over that

1:13:32

it's nicely going backwards on the equation so we have the Lambda values

1:13:37

then we pass them to Value moments which is an interesting class here that I have and this is this is the exponential

1:13:44

moving average trick from the from the paper that would just estimate the range of values based on 95th and 5ifth

1:13:51

percentile with 99 DEC rate

1:13:57

so this is all this we just estimate the the scale of it to return the the

1:14:03

inverse scale that we can scale the Lambda values and values with okay and

1:14:09

we have one less Lambda value one fewer compared to values so we ignore the last

1:14:15

last value because on the last value if if if I didn't have it like this on the

1:14:20

last value they would match right they exactly match so so it

1:14:26

will be kind of pointless here so I just ignore the time step and then we have the actor loss finally we maximize the

1:14:34

probability of disadvantages according to disadvantages so if Advantage is

1:14:40

positive we increase the probability we want to increase the probability of

1:14:45

taking that action this is of course detached because uh actor

1:14:53

is because yes because we have a stop gradient here

1:14:58

right and this is the entropy term this is the uh the coefficient times entropy

1:15:05

coefficient times entropy and we step the optimizer then we optimize the

1:15:10

critic on full states that is the distribution we maximize the probability

1:15:17

of matching the Lambda value we step the optimizer we get the

1:15:22

Matrix wow we got through this all righty I

1:15:28

mean okay I didn't use the jup notebook as much as I wanted to but we got through this these are these

1:15:35

functions and that's how it trains get the data Train The W wallet train the

1:15:41

behavior is that somewhat clear my dear students any

1:15:47

questions let me know and we move on to

## Specific Neural Networks

1:15:53

networks yeah okay let's get through the neural networks our building blocks of our

1:15:59

dreamer uh so recruit model takes in recurrent State latent States and action

1:16:08

this is an exception that it's all separate because we have to process the

1:16:13

parts in different ways we want to pass all these inputs through the recurrent Network to get the new recurrent state

1:16:21

but for that we take the the old inputs not recurrent ones concatenate them

1:16:28

together lat States in action pass them through the linear layer one layer could

1:16:34

be more uh we're using only one then we pass it to activation specified in the

1:16:41

config right here's the where here's the config also this the say is specified

1:16:49

here U to the for the line linear Network and then we pass

1:16:55

all these processed lat state in action alongside recurent state to our recurent

1:17:02

Network it looks maybe a bit convoluted um that it's just in one line but I feel

1:17:08

like it's very clear I mean if you st recur networks this should be uh this

1:17:14

should be fairly simple I actually have um I actually had to learn recr networks

1:17:20

um wait where's my profile I have my recr neural network study

1:17:25

that I mean this is uh new network by hand in naai then the same one in in P

1:17:32

so if you don't know how it works I recommend you check it out uh why not

1:17:38

but this is uh this is fairly simple like based on this input and previous recurrence state we

1:17:46

get the new recurrence state next up we have our prior net and posterior net which are pretty much the

1:17:54

same net NS except they have different input right like okay actually the code

1:18:00

looks the same here exactly because we we just have input size we process the input size actually so yeah this is the

1:18:08

same but here in the initialization the input sizes pass are different

1:18:17

because yeah uh for pet we have input size recurrent size but for posterior

1:18:23

net we have input size of recurrent size plus an coded size right but the rest is the same so I

1:18:33

I feel like it's pretty neat that I organized like this that I don't have to change the internals but it's like on

1:18:39

the outside it's visible what uh what differs here so these are the Nets let's

1:18:44

go to the prior net uh and we pretty much create the um theay in size just to

1:18:51

have it neatly accessible then multiply two values uh to have the total size of

1:18:57

uh of the output then we have our multi-layer perception actually I should just call

1:19:03

it Network why is it why is like this but

1:19:08

okay it's fine I'll leave it for now so you don't have any differences between um between the comet and this tutorial

1:19:16

but yeah this is just um MLP from this function that just that creates the

1:19:21

sequential Network that's quite easy to read without getting into this um this function it just append the the layers

1:19:29

we have the input size let me pass the list with specified sizes of hidden

1:19:36

layers and the size of the list determines how many layers there are so that's quite simple we just have like

1:19:44

list of one element of specified hidden size multiplied by the number of layers

1:19:50

that we specify in the configs so I mean this should be pretty simple that's output that's activation okay so that's

1:19:57

just like a simple multi-layer perception so in this case our output size is just 16 by

1:20:04

16 all right we pass in the loges through the through the network we arrange them in this Matrix

1:20:14

to have uh 16 distributions from that soft marks this trick is applying

1:20:21

uni like mixing uniform distribution with what

1:20:27

the network outputs and it's a trick from the from the from the paper oh here

1:20:33

it is so they observed spikes in K loss and to prevent they just mix in 1% of

1:20:41

uniform distribution which is like does it even change the performance

1:20:47

like why do we need to prevent the spikes like I observe the spikes actually without without the unimix

1:20:53

but I why you do it I just do it because that's what they wrote in the paper so I

1:21:00

mean as as you tracing what's What's Happening Here we create ones divided by

1:21:05

number of classes to like evenly distribute their probabilities we mix it

1:21:11

the probabilities with the uniform distribution and new logits are passed to our distribution to create the

1:21:18

distributions of like one hot um one hot output for each of the rows of this

1:21:26

representation right I hope that's clear I mean for clarity if you want to study

1:21:31

it you should you can delete all these right the uniform mix doesn't really matter we just output logs here like

1:21:39

this is the whole um the whole function we create the distributions sample from it with

1:21:45

reparameterization trick so I sample from it so the gradients can flow through it and then we um for the sample

1:21:54

for the output we collapse the um we collapse the last two Dimensions because

1:22:00

we need we don't need to have them eared in The Matrix but we keep the Logics uh

1:22:06

for further processing and I mean that should be pretty simple if you um if you

1:22:11

analyze it further re model simple neuron Network very very simple with two

1:22:18

outputs of mean and log standard deviation then reward model with only

1:22:24

two outputs so we pass the input through the network split it in

1:22:30

two along the last Dimension to get our mean and log standard

1:22:36

deviation that we exponentiate to get the standard deviation and then we

1:22:42

create the normal distribution right so reward model as you remember Returns the

1:22:50

distribution that we just sample log proper form and pretty much the same

1:22:55

network is used for the critic and this works fantastic right we just assume

1:23:03

that the distribution of critic of like values and um rewards is normal normal

1:23:11

distribution but many like not many but environments can be more complex they

1:23:17

might have multiple modes of of rewards for example like like there's it's not

1:23:23

just go distribution maybe there are like two peaks maybe the reality is really complex and to model this in

1:23:31

actual original gmer they use categories they use like bins that represent

1:23:38

different ranges of values and they represent continuous values with these

1:23:44

discit bins that can model more complex distributions and to hot flws and like

1:23:52

this is too hard I will not explain it right now I'm studying it currently but just know that that um that this doesn't

1:23:58

quite match um the dreamer this works fantastic but like they use the the two

1:24:06

hot losss these are the bins from minus 20 to 20 in log scale so then then

1:24:14

that's like exponentiated later on but all you have to know is that we model it as normal

1:24:21

distribution because that's the most popular like distribution of the rewards it works fantastic but if you want to

1:24:28

create truly Universal algorithm like they had to use the two hot loss that I

1:24:34

don't cly support I'm working on it I implemented it but uh it doesn't like it

1:24:40

doesn't quite work like I wrote everything correctly and it doesn't work yet so um so I'm working on it but on

1:24:47

this version we have only simple normal distribution and this detail is missing

1:24:54

that that was lost so um so that's that then we have

1:25:00

the continue model simple new network one output only with beri distribution

1:25:08

and beri distribution is just either zero or one at certain probabilities

1:25:14

then we have the encoder and encoder like for now I only support image observations uh for numerical

1:25:21

observations it will be just a simple multi layer
