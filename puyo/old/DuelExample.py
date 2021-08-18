from Duel import *
import time

'''
python ctypes example

flow
() is optional
Duel_new->(Duel_reset)->Duel_run->Duel_info->Duel_input
loop: Duel_run->Duel_info->Duel_input until gameover 1p or 2p
if end do DuelReset
'''

# Need "chcp 65001" for print
d = Duel(seed=2)
# d.reset(0)
ai1, ai2 = 2, 2

# Example Gameinfo data (My | Opp)
# "0" 1p based info, "1" 2p based info (for my|opp Data)
'''
gi = d.getGameInfo(0)
print("my field: 13*6array index 0 is left-top puyo")
print(d.getMyField(gi))  # 13*6array index 0 is left-top puyo
print("opp field: 13*6array index 0 is left-top puyo")
print(d.getOppField(gi))  # 13*6array index 0 is left-top puyo
print()
print("my14remove: #true = empty")
print(d.GameInfoGetMy14remove(gi))
print("opp14remove: #true = empty")
print(d.GameInfoGetOpp14remove(gi))
print()
print("myAllclear")
print(d.GameInfoGetMyAllClear(gi))
print("oppAllclear")
print(d.GameInfoGetOppAllClear(gi))
print()
print("myNext: next 0 top bottom, next 1 top bottom")
print(d.GameInfoGetMyNext(gi))
print("oppNext: next 0 top bottom, next 1 top bottom")
print(d.GameInfoGetOppNext(gi))
print()
print("myOjama")
print(d.GameInfoGetMyOjama(gi))
print("OppOjama")
print(d.GameInfoGetOppOjama(gi))
print()
print("myEventFrame")
print(d.GameInfoGetMyEventFrame(gi))
print("OppEventFrame")
print(d.GameInfoGetOppEventFrame(gi))
print()

# Input test (test for possible to place)
print("input test 1p input0(0column no rotaion drop): if True, able to place")
print(d.inputTest(0, 0))
print("input test 2p input0(0column no rotaion drop): if True, able to place")
print(d.inputTest(1, 0))

# Next test
gi = d.getGameInfo(0)
print("myNext: next 0 top bottom, next 1 top bottom")
print(d.GameInfoGetMyNext(gi))
d.input(1, 1)
d.run()  # apply input
print("myNext: next 0 top bottom, next 1 top bottom")

# Vs example
gi = d.getGameInfo(0)
print(d.GameInfoGetMyNext(gi))
for loop in range(8):
    # first loop: get state and input
    # second loop: get state with before loop's input
    state = d.run()
    if(state == 0):  # 1p think
        # d.getGameInfo(0)
        print("1p need think")
    elif(state == 1):  # 2p think
        # d.getGameInfo(1)...
        print("2p need think")
    elif(state == 2):  # 1p 2p both think
        print("1p, 2p need think")
        # d.getGameInfo(1)...
        # d.getGameInfo(0)...
    elif(state >= 3):  # game end
        print("game end")

    # if 1p input unnessary, set ai1 value to -1 or any
    # if 2p input unnessary, set ai2 value to -1 or any
    d.input(ai1, ai2)
    print("think succeed")

# Copy Example
print("--처음 생성한 d의 화면 출력--")
d.print()

print("--새로 생성한 d2의 화면--")
d2 = Duel(seed=2)  # other new duel object
d2.print()

print("--d를 복제한 d3의 화면--")
d3 = Duel(duel=d)  # copy duel object "d" into "d3"
d3.print()

# Ojama posiblity test
d.reset()
d.runWithOjamaSim(playerNum=0)
while(True):  # 1p will get 1ojama
    state = d.runWithOjamaSim()
    if (state == 6):
        break
    d.input(12, 10)
d.print()
OjamaArr = d.getOjamaSim()
print(type(OjamaArr[0]))
for i in OjamaArr:
    i.print()


d.print()
'''
for i in range(6):
    d.input(4, 4)
    d.run()
d.print()
print(d.GetValidMoves())
