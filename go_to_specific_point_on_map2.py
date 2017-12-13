#!/usr/bin/env python

import rospy  #needed for general running
import numpy as np #needed for going to a point
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal #needed to go to a point
import actionlib #needed 
from actionlib_msgs.msg import * #needed
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3 #unsure if needed as its imported next but error thrown it not
import geometry_msgs.msg
import scipy
from scipy.spatial import distance

import random #needed for random search
import tf #needed for a tf listener
import tf2_ros #needed for a tf broadcaster
from sensor_msgs.msg import Image #needed to take a snapshot
from cv_bridge import CvBridge, CvBridgeError #needed for errors
import cv2 #needed to save photos


from new_feature_matching_final3 import identify_image

import mapping

# MY FUNCTIONS

def GoToPoint(x,y,theta):
	position = {'x': x, 'y' : y}
	quaternion = {'r1' : 0.000, 'r2' : 0.000, 'r3' : np.sin(theta/2.0), 'r4' : np.cos(theta/2.0)}
	
	rospy.loginfo("Go to (%s, %s, %s) pose", position['x'], position['y'], theta)
	success = navigator.goto(position, quaternion)
	
	return success


class GoToPose():
    def __init__(self):

        self.goal_sent = False

	# What to do if shut down (e.g. Ctrl-C or failure)
	rospy.on_shutdown(self.shutdown)
	
	# Tell the action client that we want to spin a thread by default
	self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
	rospy.loginfo("Wait for the action server to come up")

	# Allow up to 5 seconds for the action server to come up
	self.move_base.wait_for_server(rospy.Duration(5))

    def goto(self, pos, quat):

        # Send a goal
        self.goal_sent = True
	goal = MoveBaseGoal()
	goal.target_pose.header.frame_id = 'map'
	goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = Pose(Point(pos['x'], pos['y'], 0.000),
                                     Quaternion(quat['r1'], quat['r2'], quat['r3'], quat['r4']))

	# Start moving
        self.move_base.send_goal(goal)

	# Allow TurtleBot up to 60 seconds to complete task
	success = self.move_base.wait_for_result(rospy.Duration(60)) 

        state = self.move_base.get_state()
        result = False

        if success and state == GoalStatus.SUCCEEDED:
            # We made it!
            result = True
        else:
            self.move_base.cancel_goal()

        self.goal_sent = False
        return result

    def shutdown(self):
        if self.goal_sent:
            self.move_base.cancel_goal()
        rospy.loginfo("Stop")
        rospy.sleep(1)

class snapshot():
	
	flag = False
	Snapshot_num = 0
	
	def __init__(self):
		
		rospy.Subscriber('camera/rgb/image_raw', Image, self.callback)
		self.cvb = CvBridge()

	def callback(self, data):
		try:
			img_rgb = self.cvb.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError:
			pass
			
		if self.flag == True:
			address = ('/home/turtlebot/catkin_ws/src/project/src/detections/image_S2-7_%s_0.png' %self.Snapshot_num)
			cv2.imwrite(address, img_rgb)
			img_rgb = identify_image(img_rgb) 
			address = ('/home/turtlebot/catkin_ws/src/project/src/detections/image_S2-7_%s_1.png' %self.Snapshot_num)
			cv2.imwrite(address, img_rgb)
			self.Snapshot_num += 1
			self.flag = False
			
		


def callback(self, data):
	img_rgb = self.cvb.imgmsg_to_cv2(data, "bgr8")
	cv2.namedWindow('Camera_Feed')
	cv2.imgshow('Camera_Feed', img_rgb)
	cv2.imwrite('weapon_snapshot_colmustard.png', img_rgb)

if __name__ == '__main__':
	try:
		rospy.init_node('nav_test', anonymous=False)
		
		navigator = GoToPose() #navigator object
		desired_velocity = Twist() #Object to drive with
		Snapshot = snapshot() #object of snapshot
		
		Broadcaster = tf2_ros.StaticTransformBroadcaster() #Broadcaster object
		listener = tf.TransformListener() # Create a tf listener		

		rospy.sleep(3) # Sleeping for 3 second to listen to tf messages.
		
		mapInfo = mapping.mappingFunc("/home/turtlebot/catkin_ws/src/project/src/map.yaml")#uses the mapping.py script to get map info
		
		print mapInfo[0]
		
		Left = mapInfo[2][0] #most left point
		Right = mapInfo[2][0] + (mapInfo[0][1]*mapInfo[1])  # most right point
		Top = mapInfo[2][1] + (mapInfo[0][3]*mapInfo[1])
		Bottom = mapInfo[2][1]

		print Left, Right, Top, Bottom
		
        #Left = 0.5 #most left point
        #Right = 2.5 # most
        #Top = 0.8
        #Bottom = 0.1
		
		#number of points to be randomly generated
        	numberOfPoints = 15


		trans = 0
		rot = 0
		
		ar_coordinates = []

		Previous_Ar_Marker = (100,100,100),(100,100,100,100)

		while(Snapshot.Snapshot_num <= 1):
			#generating points
			points = [[round(random.uniform(Left,Right),3), round(random.uniform(Bottom, Top),3)] for x in range(numberOfPoints)]
	    
			#center point
            #cx = (Left+Right)/2
            #cy = (Top+Bottom)/2

			#ENTER GIVEN COORDINATES HERE
			cx =
			cy = 

			points[0] = [cx,cy]		
		
			dis = distance.cdist(points, points, 'euclidean') #distance matrix
		
			visited = [] #Serves as a list of points in the order the robot should follow
	    		unvisited = points #To avoid any unnecessary computations for the points already visited


	    		visited.append(points[0]) #First point to be visited is the first one generated
	    		unvisited[0] = '' #Removing the visited point from the list keeping track of the unvisited ones

	    		remainingPoints = numberOfPoints - 1
	    		j = 0

	    		#Finding the next closest point after visiting each point
	    		while (remainingPoints > 5):

				closest = min((x for x in range(numberOfPoints) if unvisited[x]), key = lambda x:dis[j][x])
				j = closest

				visited.append(points[closest])
				unvisited[closest] = ''

				remainingPoints -= 1
			print visited

			for i in range(numberOfPoints-5):
		    	
				if(Snapshot.Snapshot_num > 1):
					break

				success = GoToPoint(visited[i][0],visited[i][1],round(random.uniform(1,-1),3))
			
				try:
					(trans,rot) = listener.lookupTransform('/map', '/ar_marker_0', rospy.Time(0)) #sees if it saw a AR Marker

				except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException): #exspetion if it didnt
					trans = 0,0,0
					rot = 0,0,0,0
					pass
			
			
				# if you have found a marker
				if(trans[0] != 0):
					if(Snapshot.Snapshot_num > 1):
						break
					# Is the new marker in the place as the last marker
					# is the new x between the boundary of +-0.4 from the old one                                     is the new y between the boundary of +- from the old one

					print(trans)
					print(Previous_Ar_Marker)
					
					if(((trans[0] > (Previous_Ar_Marker[0][0]-0.5)) and (trans[0] < (Previous_Ar_Marker[0][0]+0.5))) and ((trans[1] > (Previous_Ar_Marker[0][1]-0.5)) and (trans[1] < (Previous_Ar_Marker[0][1]+0.5)))):
					
						print("Same Marker")
				
					else:
						# make a marker 0.5 away from the x of the ar_Marker	
						static_transformStamped = geometry_msgs.msg.TransformStamped()   #object
		
						static_transformStamped.header.stamp = rospy.Time.now()    
						static_transformStamped.header.frame_id = "ar_marker_0"    #parent
						static_transformStamped.child_frame_id = 'picture_marker'  #child

						static_transformStamped.transform.translation.x = 0.0      #translations
						static_transformStamped.transform.translation.y = 0.0
						static_transformStamped.transform.translation.z = 0.4
			 
						static_transformStamped.transform.rotation.x = 0.0         #rotations
						static_transformStamped.transform.rotation.y = 0.0
						static_transformStamped.transform.rotation.z = 0.0
						static_transformStamped.transform.rotation.w = 1.0
			
						Broadcaster.sendTransform(static_transformStamped)
					
						## GO TO POINT
						while(1): #loops is here because sometimes the ar_marker isnt created in time for it to be read
							try:
								(point_translation, point_rotation) = listener.lookupTransform('/map', '/picture_marker', rospy.Time(0)) #read where the new marker is in the map
								break
							except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException): #exspetion if it didnt
								pass
					
						#test prints	
						#print(trans)
						#print(point_translation)
						#print("   ")
						#print(rot)
						#print(point_rotation)
					
						success = GoToPoint(point_translation[0],point_translation[1], rot[2]) 
					
						if (success):
						
						
							pub = rospy.Publisher('mobile_base/commands/velocity', Twist, queue_size = 10) #sets up the publisher to drive the turtlebot
						
							while(1):
						
								try: 
									Marker_translation, Marker_Rotation = listener.lookupTransform('/camera_depth_optical_frame', '/ar_marker_0', rospy.Time(0)) #where is marker relative to robot
								except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException): #exspetion if it didnt
									Marker_Rotation = 0,0,0,1
									pass

								print(Marker_Rotation[3])
							

								if((Marker_Rotation[3] > 0.1) or (Marker_Rotation[3] < -0.1)): # if not head on spin
									desired_velocity.angular.z = 0.2 #sets angular velocity
									pub.publish(desired_velocity) #publshes velocity to listeners
									rospy.sleep(0.5) #small sleep so the system isnt spammed
								else:
									try:
										test_trans_1, test_rot_1 = listener.lookupTransform('/camera_depth_optical_frame', '/ar_marker_0', rospy.Time(0)) #Gets Ar_Marker_0 coords
									except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException): 
										pass
								
									rospy.sleep(0.2) # waits so values have time to update
									try:
										test_trans_2, test_rot_2 = listener.lookupTransform('/camera_depth_optical_frame', '/ar_marker_0', rospy.Time(0)) #Gets Ar_Marker_0 coords
									except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException): 
										pass
									print(test_trans_1[0], test_trans_2[0])
									if(((test_trans_1[0] != test_trans_2[0]) and (test_trans_1[1] != test_trans_2[1])) and (test_trans_2[0] < 1.25)): #if they are the same and Ar_Marker is not in view as it hasn't updated and if the AR_marker is close									
										try:
											(ar_trans,ar_rot) = listener.lookupTransform('/map', '/ar_marker_0', rospy.Time(0)) 
										except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException): 
											pass

										rospy.sleep(1) #sleeps so turtlebot isnt moving
									
										print("correct image check")
										print(trans[0], ar_trans[0])
										print(trans[1], ar_trans[1])

										if((ar_trans[0] > trans[0]-0.3) and (ar_trans[0] < trans[0]+0.3) and (ar_trans[1] > trans[1]-0.3) and (ar_trans[1] < trans[1]+0.3)):
								
											if(Snapshot.Snapshot_num <= 1):
												Snapshot.flag = True #take a snapshot
												Previous_Ar_Marker = trans, rot #sets the marker so it doesnt find the same one again
												ar_coordinates.append((trans[0],trans[1]))
											
											break
										else:
											print("could not see AR_Marker")
											for i in range(10):
												desired_velocity.angular.z = 0.2 #sets angular velocity
												pub.publish(desired_velocity) #publshes velocity to listeners
												rospy.sleep(0.5) #small sleep so the system isnt spammed


									else:
										print("different marker")
										for i in range(30):
											desired_velocity.angular.z = 0.2 #sets angular velocity
											pub.publish(desired_velocity) #publshes velocity to listeners
											rospy.sleep(0.1) #small sleep so the system isnt spammed
									
							# Set Trans and Rot = 0 as to not accidently call if		
							trans = 0
							rot = 0
						
							rospy.sleep(4)

			
				if success:
					rospy.loginfo("Hooray, reached the desired pose")
				else:
					rospy.loginfo("The base failed to reach the desired pose")

			

				# Sleep to give the last log messages time to be sent
				rospy.sleep(1)

		print("ending")
		text_file = open("/home/turtlebot/catkin_ws/src/project/src/Coordinates.txt", "w")
		for t in ar_coordinates:
			text_file.write(' '.join(str(s) for s in t) + '\n')
		text_file.close()
		print("ended")
		rospy.spin()

	except rospy.ROSInterruptException:
			rospy.loginfo("Ctrl-C caught. Quitting")

	

	

