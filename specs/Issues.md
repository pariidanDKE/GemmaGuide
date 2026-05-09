
## 1

- When asked a relatively straightforward navigation question (how to get to my suitcase without objects in the way), the model is very dumb, it cannot safly instruct the person to walk around a table and chairts, it litreally says that you can walk straight ahead, when there is a visible table and chair blocking the way.  When I asked it a follow up quesiton on how do I get there without running into something. The model sent 4 measure object requests and got back distances. It reported them back, but whas unable to  comprhened that they are staying in the way of the suitcase I intially asked whetrehr I could pick up. It told me I could just walk straight ahead when that was clearly not the case.


Possible solutions:
- Introduce a condition to it's thinking, underline that it is a blind perosn, and that they would walk to that object, if the object that they desire has other objects in the way which are closer, and in the same direction, then those objects are obstacles, the person cannot walk thourhg them. They have to go around them. YOu can find the exact sitance to the obstacle, and then tell the user to walk around that....., 


- Perhaps a solution can be to  Have another gemma assistant tasked abvout reasoning about this specific thing. They would be a navigation agent, the task of one agent would be to draw boxes of the target object and otehr obstalces standing in the way, the task of another agent would be navaigation around said boxes


