# Task 2 comparison - `custom_20`

Protocol A (greedy / zero temperature), first sentence of each continuation, with its final label in brackets.

## Factual

| Prompt | Expected | ours: TinyStories-1M | ours: TinyStories-8M | ours: TinyStories-33M | ours: TinyStories-1Layer-21M |
|---|---|---|---|---|---|
| Tom left his ice cream outside in the hot sun. After a little whi… | Says it melted / started to melt or… | melt. **[correct]** | melt. **[correct]** | melt. **[correct]** | melt. **[correct]** |
| The room was dark, so Lucy wanted to make it bright. She walked o… | A light, lamp, or light switch. | door. **[incorrect]** | light. **[correct]** | light. **[correct]** | light. **[correct]** |
| Ben ran and ran until he was very thirsty. He went inside and dra… | A drink: water, milk, or juice. | water. **[correct]** | water. **[correct]** | water. **[correct]** | water. **[correct]** |
| The little duck lived beside a pond. When she wanted to get to th… | Says she swam / began to swim (padd… | swim around. **[correct]** | struggle. **[partial]** | swim. **[correct]** | swim. **[correct]** |
| Mia looked up at the sky late at night. She saw the big round moo… | Stars. | clouds. **[incorrect]** | stars. **[correct]** | stars. **[correct]** | stars. **[correct]** |
| Dad looked out and said it was raining hard. Before Sam went outs… | Umbrella, raincoat, or boots. | shoes and ran outside. **[incorrect]** | toys and put them in the bin. **[incorrect]** | umbrella and opened it. **[correct]** | umbrella and opened it. **[correct]** |
| It got very cold in the winter. The water in the pond outside slo… | Ice (frozen / froze is acceptable). | the water. **[incorrect]** | ash. **[incorrect]** | ice. **[correct]** | a big puddle. **[incorrect]** |

## Reasoning

| Prompt | Expected | ours: TinyStories-1M | ours: TinyStories-8M | ours: TinyStories-33M | ours: TinyStories-1Layer-21M |
|---|---|---|---|---|---|
| Ben forgot to close the window before the big storm. When he came… | Wet (or covered in water / soaked). | gone. **[incorrect]** | wet. **[correct]** | wet and slippery. **[correct]** | wet. **[correct]** |
| There were three hats on the shelf: a red one, a blue one, and a … | The blue hat. | blue one. **[correct]** | green one. **[incorrect]** | green hat. **[incorrect]** | red one. **[incorrect]** |
| Sara had only one coin. The apple cost one coin and the big toy c… | The apple (the only affordable item… | coin. **[incorrect]** | apple. **[correct]** | big toy. **[incorrect]** | toy. **[incorrect]** |
| Every morning Max brushed his teeth first and ate after that. Thi… | Breakfast (or a breakfast food). | favorite snack. **[partial]** | breakfast. **[correct]** | breakfast. **[correct]** | breakfast. **[correct]** |
| Lily was carrying a full glass of milk much too fast, and then sh… | Spilled / fell / went everywhere. | was very hot and it was too hot. **[incorrect]** | spilled all over the floor and made a b… **[correct]** | spilled all over her dress and the floo… **[correct]** | spilled all over the floor and made a b… **[correct]** |
| It was freezing outside. Tom had a thin coat and a thick coat. He… | The thick coat. | coat and coat. **[partial]** | coat and went outside. **[partial]** | coat and the thin coat. **[partial]** | coat. **[partial]** |
| Pip the puppy was very afraid of deep water. His friends ran to t… | A refusal: no, or that he is scared… | , "I'm scared of you, I'm scared of you. **[incorrect]** | yes and jumped in. **[incorrect]** | , "No, I don't want to. **[correct]** | , "I can't swim, I'm too scared." **[correct]** |

## Context

| Prompt | Expected | ours: TinyStories-1M | ours: TinyStories-8M | ours: TinyStories-33M | ours: TinyStories-1Layer-21M |
|---|---|---|---|---|---|
| Nina and her brother Leo went to the park. Nina brought a red bal… | The kite (blue kite). | kite. **[correct]** | kite. **[correct]** | kite. **[correct]** | kite. **[correct]** |
| Sam had a toy car and his sister Ella had a toy train. Sam gave h… | The toy car. | toy car. **[correct]** | toy train. **[incorrect]** | toy train. **[incorrect]** | toy car. **[correct]** |
| Mom put the cookies in a tin on the top shelf, and she put the br… | The tin / the top shelf (not the ta… | table. **[incorrect]** | shelf. **[correct]** | shelf and tried to reach the cookie. **[correct]** | table. **[incorrect]** |
| The cup on the table was full of juice. Rosa picked it up and dra… | Empty. | very sweet and yummy. **[incorrect]** | full and it was time for a nap. **[incorrect]** | empty. **[correct]** | full of juice. **[incorrect]** |
| On the farm there were three animals: a cow named Bella, a pig na… | Dot (the duck). | very happy. **[incorrect]** | very loud. **[incorrect]** | the perfect match. **[incorrect]** | very nice. **[incorrect]** |
| Jon put on his yellow coat and his friend Ana put on her purple c… | His yellow coat (yellow, not purple… | coat. **[partial]** | friend. **[incorrect]** | coat anywhere. **[partial]** | coat. **[partial]** |
