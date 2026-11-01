# -*- coding: utf-8 -*-
"""V14: independently computed, deterministic, parameterized multi-step tasks.
20 families/domain, 10 variants per family; no model-outcome-based filtering.
"""
import random, json, math, hashlib
from pathlib import Path
SEED=20261031
CATEGORIES=('control','boundary','transition','recovery')

def puzzle(domain, family, r):
    typ=family%10
    if domain=='logic':
        if typ==0: # chained implication and parity
            n=r.randint(9,15); start=r.randrange(2); flips=[r.randrange(2) for _ in range(n)]; q=f'A binary signal begins at {start}. Passing {n} gates, each gate flips the bit iff its flag is 1. Flags: {flips}. After all gates, what is the bit? Answer integer.'; a=start^(sum(flips)%2)
        elif typ==1: # ordered set union intersection
            a=set(r.sample(range(1,41),14)); b=set(r.sample(range(1,41),15)); c=set(r.sample(range(1,41),12)); op=r.choice(['union','intersection']); target=(a|b)-c if op=='union' else (a&b)-c; q=f'A={sorted(a)}; B={sorted(b)}; C={sorted(c)}. Form {"(A union B) minus C" if op=="union" else "(A intersection B) minus C"}. How many distinct elements remain? Answer integer.'; a=len(target)
        elif typ==2: # robust order
            names=list('ABCDEFGH'); r.shuffle(names); cuts=[r.randint(2,20) for _ in range(7)]; key=r.randrange(1,8); q='; '.join(f'{names[i]} is {cuts[i]} minutes before {names[i+1]}' for i in range(7)); q=f'Events form this linked timeline: {q}. Which event is {key+1}th chronologically? Answer one letter.';a=names[key]
        elif typ==3: # constrained path counting
            a0,b,c=[r.randint(3,7) for _ in range(3)];ban=r.randint(1,a0-1); q=f'A layered graph has {a0} distinct first choices, {b} distinct second choices per first choice and {c} distinct final choices per second choice. Exactly {ban} first-choice branches are entirely prohibited. Each remaining second-choice node loses precisely one of its {c} outgoing choices. Count valid paths. Answer integer.'; a=(a0-ban)*b*(c-1)
        elif typ==4: # temporal constraints with nonlocal removal
            vals=r.sample(range(15,130),9); k=r.randint(3,6); ranked=sorted(vals,reverse=True); q=f'Task completion times are {vals} minutes. Exclude the shortest {k} tasks, then exclude the single longest of the remaining tasks. What is the longest remaining time? Answer integer.';a=ranked[1]
        elif typ==5: # modular conditional boolean
            x=r.randint(40,230); p=r.randint(3,12); m=r.randint(3,9); t=r.randint(2,6); q=f'A system opens iff (({x} mod {p}) < {t}) AND (({x} mod {m}) is even OR {x} > 150). Is it open? Answer TRUE or FALSE.';a='TRUE' if (x%p<t and ((x%m)%2==0 or x>150)) else 'FALSE'
        elif typ==6: # cyclic action windows
            n=r.randint(11,20); arr=[r.randint(-8,10) for _ in range(n)]; initial=r.randint(15,40); l=r.randint(2,5);u=r.randint(l+4,n); q=f'A register starts at {initial}. Apply signed updates {arr} in order, but skip updates numbered {l} through {u} inclusive (1-indexed). What is the final value? Answer integer.';a=initial+sum(v for i,v in enumerate(arr,1) if not l<=i<=u)
        elif typ==7: # condition cascading
            vals=r.sample(range(100,900),11); threshold=r.randint(220,380); kept=sorted((v for v in vals if v>=threshold),reverse=True); k=r.randint(1,min(4,len(kept)-1));q=f'Scores: {vals}. Remove scores below {threshold}, then remove the highest surviving score. What is the {k}th highest among survivors? Answer integer.';a=kept[k]
        elif typ==8: # dynamic filter and order
            vals=r.sample(range(1,55),12); even=[x for x in vals if x%2==0]; k=r.randint(1,min(4,len(even)));q=f'The sequence is {vals}. Retain only even entries; sort them in increasing order. What is the {k}th retained value? Answer integer.';a=sorted(even)[k-1]
        else: # boolean logical formula
            x,y,z=[r.randrange(2) for _ in range(3)]; flags=[r.randrange(2) for _ in range(6)];v=x;v^=flags[0];v|=y;v^=flags[1];v&=z|flags[2];v^=sum(flags[3:])%2;q=f'Binary state X={x}, Y={y}, Z={z}. Apply in order: XOR {flags[0]}, OR Y, XOR {flags[1]}, AND (Z OR {flags[2]}), XOR {sum(flags[3:])%2}. What is the final bit? Answer integer.';a=v
    else:
        if typ==0: # CRT via construction
            p,q=r.sample([7,9,11,13,17],2); n=r.randint(100,800); a1=n%p;b1=n%q; smallest=next(i for i in range(p*q) if i%p==a1 and i%q==b1);q=f'Find the least nonnegative integer x satisfying x ≡ {a1} (mod {p}) and x ≡ {b1} (mod {q}). Answer integer.';a=smallest
        elif typ==1: # two-stage replacement
            vol=r.choice([400,800,1200]);p=r.choice([10,20,25,30]);s=r.choice([10,20,25,30]);a=vol*(100-p)*(100-s)//10000;q=f'A {vol}-liter tank contains pure dye. Replace {p}% of the mixed contents with water, mix thoroughly, then replace another {s}% with water. How many liters of dye remain? Answer integer.'
        elif typ==2: # constrained combinatorics
            n=r.randint(12,22);k=r.randint(4,6);forbidden=r.randint(3,5);a=math.comb(n-forbidden,k)+forbidden*math.comb(n-forbidden,k-1);q=f'Choose {k} people from {n}. Among them {forbidden} specified people are mutually exclusive: at most one of these {forbidden} may be chosen. How many committees are valid? Answer integer.'
        elif typ==3: # equations then expression
            x=r.randint(10,50);y=r.randint(10,40);p=r.randint(2,7);v=r.randint(2,7);a=r.randint(2,6)*x+r.randint(2,6)*y;cx=r.randint(2,6);cy=r.randint(2,6);a=cx*x+cy*y;q=f'Solve x+y={x+y} and {p}x-{v}y={p*x-v*y}. Find {cx}x+{cy}y. Answer integer.'
        elif typ==4: # recurrence
            a0=r.randint(3,18);d=r.randint(2,9);dd=r.randint(1,5);n=r.randint(7,13); seq=[a0];step=d
            for _ in range(1,n):seq.append(seq[-1]+step);step+=dd
            a=seq[-1];q=f'An integer sequence begins at {a0}. The increment from term 1 to term 2 is {d}, and each later increment grows by {dd} compared with the preceding increment. What is term {n}? Answer integer.'
        elif typ==5: # modular + linear restriction
            m=r.randint(6,14);rem=r.randrange(m);L=r.randint(45,110);U=L+r.randint(35,80);valid=[k for k in range(L,U+1) if k%m==rem]; a=sum(valid);q=f'Find the sum of all integers x satisfying {L} ≤ x ≤ {U} and x ≡ {rem} (mod {m}). Answer integer.'
        elif typ==6: # inclusion exclusion
            n=r.randint(80,240);p=r.choice([3,4,5,6]);z=r.choice([7,8,9,11]);a=n//p+n//z-n//math.lcm(p,z);q=f'Among integers from 1 through {n} inclusive, how many are divisible by {p} OR by {z}? Answer integer.'
        elif typ==7: # weighted path count with blockage
            p=r.randint(3,8);z=r.randint(3,7);t=r.randint(2,6);block=r.randint(1,p-1);a=(p-block)*z*t;q=f'A three-stage machine has {p} independent initial choices, {z} intermediate choices per initial choice, and {t} endings per intermediate choice. Exactly {block} initial choices are disallowed. How many different complete procedures remain? Answer integer.'
        elif typ==8: # currency with discount and fee
            n=r.randint(4,10);unit=r.randint(2,9)*20;discount=r.choice([10,20,25]);fee=r.randint(8,30);a=n*unit*(100-discount)//100+fee;q=f'An order contains {n} identical items at {unit} credits each. A {discount}% discount applies to the entire item subtotal, then a fixed shipping charge of {fee} credits is added. What is the total cost? Answer integer.'
        else: # arithmetic with excluded terms
            a0=r.randint(4,20);d=r.randint(2,12);n=r.randint(12,22);excluded=r.sample(range(1,n+1),3);a=sum(a0+(i-1)*d for i in range(1,n+1) if i not in excluded);q=f'An arithmetic progression starts at {a0}, increases by {d}, and has {n} terms. Omit terms numbered {sorted(excluded)} (1-indexed). What is the sum of the remaining terms? Answer integer.'
    return q,str(a)

def make_dataset(domain,n=200):
    if domain not in ('logic','math') or not (1<=n<=200):raise ValueError('domain logic/math; n=1..200')
    out=[];seen=set()
    for i in range(n):
        fam=i%20; variant=i//20
        for attempt in range(1000):
            r=random.Random(SEED+(0 if domain=='logic' else 100000000)+fam*100000+variant*1009+attempt)
            q,a=puzzle(domain,fam,r)
            if q not in seen:break
        else:raise RuntimeError('Could not generate unique item')
        seen.add(q)
        group=CATEGORIES[fam//5]
        out.append(dict(id=i+1,family=f'{domain.upper()}_{fam+1:02d}',mechanism=group,difficulty='designed_multi_step',variant=variant+1,question=q,answer=a))
    return out

def load_dataset_v14(domain,path,n=200):
    ds=make_dataset(domain,200)
    Path(path).write_text(json.dumps(ds,ensure_ascii=False,indent=2),encoding='utf-8')
    return ds[:n]

if __name__=='__main__':
    for d in ('logic','math'):
        x=make_dataset(d); assert len(set(z['question'] for z in x))==200
        load_dataset_v14(d,f'benchmark_v14_{d}_mechanism.json')
        print(d,len(x),{g:sum(z['mechanism']==g for z in x) for g in CATEGORIES})
