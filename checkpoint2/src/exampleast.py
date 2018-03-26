class AssassinsBlade (WeaponCard):
    def __init__(self):
        super() . __init__(" Assassin's Blade ", 5, CHARACTER_CLASS . ROGUE, CARD_RARITY . COMMON)

    def create_weapon(self, player):
        return Weapon(3, 4)
