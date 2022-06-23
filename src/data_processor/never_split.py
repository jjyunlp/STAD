"""
特殊字符
感觉可以直接从vocab.txt上得到这个信息，然后拿来用。避免xiu
"""

never_split = ["[e1]", "[/e1]", "[e2]", "[/e2]", "[E1]", "[/E1]", "[E2]", "[/E2]",  
            "[person]", "[organization]", "[city]", "[country]", "[cause_of_death]", 
            "[criminal_charge]", "[date]", "[duration]", "[location]", "[number]", 
            "[ideology]", "[title]", "[nationality]", "[religion]", "[misc]", 
            "[state_or_province]", "[url]", "[unk_type]",
            "[/person]", "[/organization]", "[/city]", "[/country]", 
            "[/cause_of_death]", "[/criminal_charge]", "[/date]", "[/duration]", 
            "[/location]", "[/number]", "[/ideology]", "[/title]", "[/nationality]",
            "[/religion]", "[/misc]", "[/state_or_province]", "[/url]",
            "[start]", "[middle]", "[end]",
            '[r1]', '[/r1]', '[r2]', '[/r2]', '[r3]', '[/r3]',
            '[subj]', '[obj]', '[mask]',
            '[sen]', '[sdp]', '[sdp_k1]', '[middle]', '[subject]', '[object]', 
            '[r]', '[/r]',
            '[head]', '[tail]', '[blank]', '[e]']

# [r] for root token
